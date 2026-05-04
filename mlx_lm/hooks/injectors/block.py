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

    # Detect which attention attribute this block uses
    if hasattr(block_module, "linear_attn"):
        attn_attr = "linear_attn"
    elif hasattr(block_module, "self_attn"):
        attn_attr = "self_attn"
    else:
        for name in ["attn", "attention"]:
            if hasattr(block_module, name):
                attn_attr = name
                break
        else:
            raise ValueError(
                f"Block at layer {layer_idx} has no recognized attention attribute."
            )

    original_class = block_module.__class__

    # Detect whether this block uses the standard single-tensor return convention
    # or a tuple return (e.g. Gemma 4 returns (h, shared_kv, offset)).
    # We probe by checking the original __call__ annotation or falling back to
    # a simpler pre/post-only wrapping for non-standard blocks.
    _has_standard_mlp = (
        hasattr(block_module, "mlp")
        and hasattr(block_module, "post_attention_layernorm")
        and hasattr(block_module, "input_layernorm")  # OLMo 3 / OLMo 2 are post-norm and lack this
        and not hasattr(block_module, "pre_feedforward_layernorm")  # Gemma 4 uses this
    )

    class HookedBlock(original_class):
        def __call__(
            self,
            x: mx.array,
            mask: Optional[mx.array] = None,
            cache: Optional[Any] = None,
            **kwargs,
        ):
            """Apply the block while emitting residual-stream hook activations."""
            context_kwargs = {'cache': cache, 'mask': mask}
            resid = hook_resid_pre(x, **context_kwargs)

            if _has_standard_mlp:
                # Standard transformer block: reimplement to capture resid_mid.
                attn_module = getattr(self, attn_attr)
                attn_out = attn_module(
                    self.input_layernorm(resid),
                    mask,
                    cache,
                    **kwargs,
                )
                h = resid + attn_out
                h = hook_resid_mid(h, **context_kwargs)
                mlp_out = self.mlp(self.post_attention_layernorm(h))
                out = h + mlp_out
                out = hook_resid_post(out, **context_kwargs)
                return out
            else:
                # Non-standard block (e.g. Gemma 4 with MoE, per-layer gating,
                # tuple returns). Delegate entirely to the original forward pass
                # and hook only the input and the hidden-state component of the
                # output.
                result = super().__call__(resid, mask, cache, **kwargs)
                if isinstance(result, tuple):
                    h = hook_resid_post(result[0], **context_kwargs)
                    return (h,) + result[1:]
                else:
                    return hook_resid_post(result, **context_kwargs)

    block_module.__class__ = HookedBlock

    return hook_points
