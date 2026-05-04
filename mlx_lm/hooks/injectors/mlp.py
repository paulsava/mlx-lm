import mlx.core as mx
import mlx.nn as nn
from typing import Any, Dict
from ..hook_point import HookPoint


def inject_mlp_hooks(
    mlp_module: nn.Module,
    layer_idx: int,
    root_module: Any
) -> Dict[str, HookPoint]:
    """Inject hook points into a Llama-style MLP module."""
    # Create hook points
    hook_points = {}

    def create_hook_point(suffix: str) -> HookPoint:
        """Helper to create and register hook point."""
        name = f"blocks.{layer_idx}.mlp.{suffix}"
        hp = HookPoint(name, base_module=None)
        hook_points[name] = hp
        root_module.register_hook_point(name, hp)
        return hp

    # Create all hook points
    hook_pre = create_hook_point("hook_pre")
    hook_gate = create_hook_point("hook_gate")
    hook_up = create_hook_point("hook_up")
    hook_act = create_hook_point("hook_act")
    hook_post = create_hook_point("hook_post")

    # Detect standard Llama-style MLP (gate_proj / up_proj / down_proj). Models with
    # a fused gate_up_proj (e.g. Phi-4) or other layouts fall back to super().__call__()
    # and only get pre/post hooks (gate/up/act will be no-ops).
    _is_standard_mlp = (
        hasattr(mlp_module, "gate_proj")
        and hasattr(mlp_module, "up_proj")
        and hasattr(mlp_module, "down_proj")
    )

    original_class = mlp_module.__class__

    class HookedMLP(original_class):
        def __call__(self, x: mx.array) -> mx.array:
            """Apply the feed-forward network while emitting hook activations."""
            x_in = hook_pre(x)
            if _is_standard_mlp:
                gate_out = self.gate_proj(x_in)
                up_out = self.up_proj(x_in)
                gate_out = hook_gate(gate_out)
                up_out = hook_up(up_out)
                act_out = nn.silu(gate_out) * up_out
                act_out = hook_act(act_out)
                output = self.down_proj(act_out)
            else:
                output = super().__call__(x_in)
            return hook_post(output)

    mlp_module.__class__ = HookedMLP

    return hook_points
