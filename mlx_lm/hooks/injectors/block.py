import mlx.core as mx
import mlx.nn as nn
from typing import Any, Optional, Dict
from ..hook_point import HookPoint


def inject_block_hooks(
    block_module: nn.Module,
    layer_idx: int,
    root_module: Any
) -> Dict[str, HookPoint]:
    """Inject hook points that cover the residual stream of a transformer block."""
    # Create hook points
    hook_points = {}

    def create_hook_point(suffix: str) -> HookPoint:
        """Helper to create and register hook point."""
        name = f"blocks.{layer_idx}.{suffix}"
        hp = HookPoint(name, base_module=None)
        hook_points[name] = hp
        root_module.register_hook_point(name, hp)
        return hp

    # Create all hook points
    hook_resid_pre = create_hook_point("hook_resid_pre")
    hook_resid_mid = create_hook_point("hook_resid_mid")
    hook_resid_post = create_hook_point("hook_resid_post")

    original_class = block_module.__class__

    class HookedBlock(original_class):
        def __call__(
            self,
            x: mx.array,
            mask: Optional[mx.array] = None,
            cache: Optional[Any] = None,
        ) -> mx.array:
            """Apply the block while emitting residual-stream hook activations."""
            context_kwargs = {'cache': cache, 'mask': mask}
            resid = hook_resid_pre(x, **context_kwargs)

            attn_out = self.self_attn(
                self.input_layernorm(resid),
                mask,
                cache
            )
            h = resid + attn_out
            h = hook_resid_mid(h, **context_kwargs)

            mlp_out = self.mlp(self.post_attention_layernorm(h))
            out = h + mlp_out
            out = hook_resid_post(out, **context_kwargs)
            return out

    block_module.__class__ = HookedBlock

    return hook_points
