import mlx.core as mx
import mlx.nn as nn
from typing import Any, Optional, Dict
from ..hook_point import HookPoint
from ..sdpa import manual_scaled_dot_product_attention


def inject_gated_attention_hooks(
    attn_module: nn.Module,
    layer_idx: int,
    root_module: Any,
    use_manual_sdpa: bool = True
) -> Dict[str, HookPoint]:
    """Inject hook points into a Qwen3.5-style gated attention module.

    Gated attention splits q_proj output into queries and a gate. After
    attention, the output is multiplied by sigmoid(gate) before o_proj.
    """
    hook_points = {}

    def create_hook_point(suffix: str) -> HookPoint:
        name = f"blocks.{layer_idx}.attn.{suffix}"
        hp = HookPoint(name, base_module=None)
        hook_points[name] = hp
        root_module.register_hook_point(name, hp)
        return hp

    hook_q = create_hook_point("hook_q")
    hook_k = create_hook_point("hook_k")
    hook_v = create_hook_point("hook_v")
    hook_q_rotated = create_hook_point("hook_q_rotated")
    hook_k_rotated = create_hook_point("hook_k_rotated")
    hook_attn_pattern = create_hook_point("hook_attn_pattern") if use_manual_sdpa else None
    hook_attn_out = create_hook_point("hook_attn_out")
    hook_z = create_hook_point("hook_z")

    original_class = attn_module.__class__

    class HookedGatedAttention(original_class):
        def __call__(
            self,
            x: mx.array,
            mask: Optional[mx.array] = None,
            cache: Optional[Any] = None,
        ) -> mx.array:
            B, L, D = x.shape

            # Q projection: output is 2x head_dim, split into queries + gate
            q_proj_output = self.q_proj(x)
            queries, gate = mx.split(
                q_proj_output.reshape(B, L, self.num_attention_heads, -1),
                2,
                axis=-1,
            )
            gate = gate.reshape(B, L, -1)

            keys = self.k_proj(x)
            values = self.v_proj(x)

            # Hook raw projections (queries already reshaped to per-head)
            queries = hook_q(queries, cache=cache, mask=mask)
            keys = hook_k(keys, cache=cache, mask=mask)
            values = hook_v(values, cache=cache, mask=mask)

            # Norms, reshape, transpose
            queries = self.q_norm(queries).transpose(0, 2, 1, 3)
            keys = self.k_norm(
                keys.reshape(B, L, self.num_key_value_heads, -1)
            ).transpose(0, 2, 1, 3)
            values = values.reshape(
                B, L, self.num_key_value_heads, -1
            ).transpose(0, 2, 1, 3)

            # RoPE + cache
            if cache is not None:
                queries = self.rope(queries, offset=cache.offset)
                keys = self.rope(keys, offset=cache.offset)
                keys, values = cache.update_and_fetch(keys, values)
            else:
                queries = self.rope(queries)
                keys = self.rope(keys)

            # Hook rotated Q/K
            queries = hook_q_rotated(queries, cache=cache, mask=mask)
            keys = hook_k_rotated(keys, cache=cache, mask=mask)

            # Scaled dot-product attention
            if use_manual_sdpa:
                def attn_pattern_hook_fn(attn_weights: mx.array) -> mx.array:
                    return hook_attn_pattern(attn_weights, cache=cache, mask=mask)

                attn_out, _ = manual_scaled_dot_product_attention(
                    queries,
                    keys,
                    values,
                    scale=self.scale,
                    mask=mask,
                    attn_pattern_hook=attn_pattern_hook_fn if hook_attn_pattern else None,
                )
            else:
                from ...models.base import scaled_dot_product_attention

                attn_out = scaled_dot_product_attention(
                    queries,
                    keys,
                    values,
                    cache=cache,
                    scale=self.scale,
                    mask=mask,
                )

            attn_out = hook_attn_out(attn_out, cache=cache, mask=mask)

            # Reshape and apply gated output projection
            output = attn_out.transpose(0, 2, 1, 3).reshape(B, L, -1)
            output = self.o_proj(output * mx.sigmoid(gate))

            output = hook_z(output, cache=cache, mask=mask)

            return output

    attn_module.__class__ = HookedGatedAttention

    return hook_points
