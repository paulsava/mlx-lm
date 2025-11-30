import mlx.core as mx
import mlx.nn as nn
from typing import Any, Optional, Dict
from ..hook_point import HookPoint
from ..sdpa import manual_scaled_dot_product_attention


def inject_attention_hooks(
    attn_module: nn.Module,
    layer_idx: int,
    root_module: Any,
    use_manual_sdpa: bool = True
) -> Dict[str, HookPoint]:
    """Inject hook points into a Llama-style attention module."""
    # Create hook points
    hook_points = {}

    def create_hook_point(suffix: str) -> HookPoint:
        """Helper to create and register hook point."""
        name = f"blocks.{layer_idx}.attn.{suffix}"
        hp = HookPoint(name, base_module=None)
        hook_points[name] = hp
        root_module.register_hook_point(name, hp)
        return hp

    # Create all hook points
    hook_q = create_hook_point("hook_q")
    hook_k = create_hook_point("hook_k")
    hook_v = create_hook_point("hook_v")
    hook_q_rotated = create_hook_point("hook_q_rotated")
    hook_k_rotated = create_hook_point("hook_k_rotated")
    hook_attn_pattern = create_hook_point("hook_attn_pattern") if use_manual_sdpa else None
    hook_attn_out = create_hook_point("hook_attn_out")
    hook_z = create_hook_point("hook_z")

    original_class = attn_module.__class__

    class HookedAttention(original_class):
        def _apply_optional_norm(self, tensor: mx.array, attr_name: str) -> mx.array:
            """Apply norm `attr_name` when the trailing dimensions match."""
            if not hasattr(self, attr_name):
                return tensor

            norm_layer = getattr(self, attr_name)
            weight = getattr(norm_layer, "weight", None)

            if weight is not None:
                weight_shape = tuple(weight.shape)
                if len(weight_shape) > 0:
                    tensor_shape = tuple(tensor.shape)
                    if len(weight_shape) > len(tensor_shape):
                        return tensor
                    tail_shape = tensor_shape[-len(weight_shape):]
                    if tail_shape != weight_shape:
                        return tensor

            return norm_layer(tensor)

        def __call__(
            self,
            x: mx.array,
            mask: Optional[mx.array] = None,
            cache: Optional[Any] = None,
        ) -> mx.array:
            """Apply attention while emitting the configured hook activations."""
            B, L, _ = x.shape

            # Q/K/V projections with hooks
            q = self.q_proj(x)
            k = self.k_proj(x)
            v = self.v_proj(x)

            # Apply hooks to projections (before reshape)
            q = hook_q(q, cache=cache, mask=mask)
            k = hook_k(k, cache=cache, mask=mask)
            v = hook_v(v, cache=cache, mask=mask)

            # Apply optional norms before reshaping (covers models like Olmo)
            q = self._apply_optional_norm(q, "q_norm")
            k = self._apply_optional_norm(k, "k_norm")

            # Reshape for multi-head attention
            q = q.reshape(B, L, self.n_heads, -1)
            k = k.reshape(B, L, self.n_kv_heads, -1)
            v = v.reshape(B, L, self.n_kv_heads, -1)

            # Apply per-head norms if defined (covers Qwen-style attention)
            q = self._apply_optional_norm(q, "q_norm").transpose(0, 2, 1, 3)
            k = self._apply_optional_norm(k, "k_norm").transpose(0, 2, 1, 3)
            v = v.transpose(0, 2, 1, 3)

            # Apply RoPE and handle cache
            if cache is not None:
                q = self.rope(q, offset=cache.offset)
                k = self.rope(k, offset=cache.offset)
                k, v = cache.update_and_fetch(k, v)
            else:
                q = self.rope(q)
                k = self.rope(k)

            # Apply hooks to rotated Q/K
            q = hook_q_rotated(q, cache=cache, mask=mask)
            k = hook_k_rotated(k, cache=cache, mask=mask)

            # Scaled dot-product attention
            if use_manual_sdpa:
                def attn_pattern_hook_fn(attn_weights: mx.array) -> mx.array:
                    """Hook to apply to attention pattern."""
                    return hook_attn_pattern(attn_weights, cache=cache, mask=mask)

                attn_out, _ = manual_scaled_dot_product_attention(
                    q,
                    k,
                    v,
                    scale=self.scale,
                    mask=mask,
                    attn_pattern_hook=attn_pattern_hook_fn if hook_attn_pattern else None,
                )
            else:
                from ...models.base import scaled_dot_product_attention

                attn_out = scaled_dot_product_attention(
                    q,
                    k,
                    v,
                    cache=cache,
                    scale=self.scale,
                    mask=mask,
                )

            # Apply hook to attention output
            attn_out = hook_attn_out(attn_out, cache=cache, mask=mask)

            # Reshape back and apply output projection
            output = attn_out.transpose(0, 2, 1, 3).reshape(B, L, -1)
            output = self.o_proj(output)

            # Apply final hook
            output = hook_z(output, cache=cache, mask=mask)

            return output

    attn_module.__class__ = HookedAttention

    return hook_points
