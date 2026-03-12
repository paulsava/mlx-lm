import mlx.core as mx
import mlx.nn as nn
from typing import Any, Optional, Dict
from ..hook_point import HookPoint


def inject_linear_attention_hooks(
    attn_module: nn.Module,
    layer_idx: int,
    root_module: Any,
) -> Dict[str, HookPoint]:
    """Inject hook points into a GatedDeltaNet linear attention module."""
    hook_points = {}

    def create_hook_point(suffix: str) -> HookPoint:
        name = f"blocks.{layer_idx}.linear_attn.{suffix}"
        hp = HookPoint(name, base_module=None)
        hook_points[name] = hp
        root_module.register_hook_point(name, hp)
        return hp

    hook_qkv = create_hook_point("hook_qkv")
    hook_conv_out = create_hook_point("hook_conv_out")
    hook_recurrence_out = create_hook_point("hook_recurrence_out")
    hook_z = create_hook_point("hook_z")

    original_class = attn_module.__class__

    class HookedLinearAttention(original_class):
        def __call__(
            self,
            inputs: mx.array,
            mask: Optional[mx.array] = None,
            cache: Optional[Any] = None,
        ) -> mx.array:
            B, S, _ = inputs.shape

            if self.sharding_group is not None:
                from mlx.nn.layers.distributed import sum_gradients
                inputs = sum_gradients(self.sharding_group)(inputs)

            qkv = self.in_proj_qkv(inputs)
            z = self.in_proj_z(inputs).reshape(B, S, self.num_v_heads, self.head_v_dim)
            b = self.in_proj_b(inputs)
            a = self.in_proj_a(inputs)

            qkv = hook_qkv(qkv, cache=cache, mask=mask)

            if cache is not None and cache[0] is not None:
                conv_state = cache[0]
            else:
                conv_state = mx.zeros(
                    (B, self.conv_kernel_size - 1, self.conv_dim),
                    dtype=inputs.dtype,
                )

            if mask is not None:
                qkv = mx.where(mask[..., None], qkv, 0)
            conv_input = mx.concatenate([conv_state, qkv], axis=1)
            if cache is not None:
                cache[0] = conv_input[:, -(self.conv_kernel_size - 1) :]
            conv_out = nn.silu(self.conv1d(conv_input))

            conv_out = hook_conv_out(conv_out, cache=cache, mask=mask)

            q, k, v = [
                t.reshape(B, S, h, d)
                for t, h, d in zip(
                    mx.split(conv_out, [self.key_dim, 2 * self.key_dim], -1),
                    [self.num_k_heads, self.num_k_heads, self.num_v_heads],
                    [self.head_k_dim, self.head_k_dim, self.head_v_dim],
                )
            ]

            state = cache[1] if cache else None
            inv_scale = k.shape[-1] ** -0.5
            q = (inv_scale**2) * mx.fast.rms_norm(q, None, 1e-6)
            k = inv_scale * mx.fast.rms_norm(k, None, 1e-6)

            from ...models.gated_delta import gated_delta_update
            out, state = gated_delta_update(
                q, k, v, a, b,
                self.A_log, self.dt_bias,
                state, mask,
                use_kernel=not self.training,
            )

            if cache is not None:
                cache[1] = state

            out = hook_recurrence_out(out, cache=cache, mask=mask)

            out = self.norm(out, z)
            out = self.out_proj(out.reshape(B, S, -1))

            if self.sharding_group is not None:
                out = mx.distributed.all_sum(out, group=self.sharding_group)

            out = hook_z(out, cache=cache, mask=mask)
            return out

    attn_module.__class__ = HookedLinearAttention

    return hook_points
