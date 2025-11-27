import mlx.nn as nn

from .hooked_module import HookedRootModule
from .injectors import inject_attention_hooks, inject_block_hooks, inject_mlp_hooks


def wrap_model_with_hooks(model: nn.Module, use_manual_sdpa: bool = True) -> nn.Module:
    """Add hook infrastructure to a loaded model in place."""
    # Mix in HookedRootModule to model class
    # This adds hook management methods without modifying model structure
    original_class = model.__class__

    class HookedModel(original_class, HookedRootModule):
        pass

    # Change model's class to the mixed version
    model.__class__ = HookedModel

    # Initialize hook infrastructure
    model._init_hooks()

    # Inject hooks into each layer
    # Access layers through model.model.layers (standard MLX-LM structure)
    if not hasattr(model, "model"):
        raise ValueError(
            "Model does not have expected structure (model.model.layers). "
            "This hook system currently only supports standard MLX-LM models."
        )

    if not hasattr(model.model, "layers"):
        raise ValueError(
            "Model does not have expected structure (model.model.layers). "
            "This hook system currently only supports standard MLX-LM models."
        )

    layers = model.model.layers

    for i, layer in enumerate(layers):
        # Inject block-level residual hooks
        inject_block_hooks(layer, i, model)

        # Detect and inject attention hooks
        # Support multiple naming conventions: self_attn, attn, attention
        attn_module = None
        for attr_name in ["self_attn", "attn", "attention"]:
            if hasattr(layer, attr_name):
                attn_module = getattr(layer, attr_name)
                break

        if attn_module is not None:
            inject_attention_hooks(attn_module, i, model, use_manual_sdpa)

        # Detect and inject MLP hooks
        # Support multiple naming conventions: mlp, feed_forward, ffn
        mlp_module = None
        for attr_name in ["mlp", "feed_forward", "ffn"]:
            if hasattr(layer, attr_name):
                mlp_module = getattr(layer, attr_name)
                break

        if mlp_module is not None:
            inject_mlp_hooks(mlp_module, i, model)

    return model


def get_hook_schema(model: nn.Module) -> dict:
    """Return metadata about every hook point registered on the model."""
    if not isinstance(model, HookedRootModule):
        raise ValueError("Model is not hooked. Call wrap_model_with_hooks() first.")

    schema = {}
    for name in model.hook_points():
        # Parse hook point name to extract metadata
        parts = name.split(".")

        if "attn" in name:
            module_type = "attention"
        elif "mlp" in name:
            module_type = "mlp"
        elif "resid" in name:
            module_type = "block"
        else:
            module_type = "unknown"

        layer_idx = None
        if len(parts) >= 2 and parts[0] == "blocks":
            try:
                layer_idx = int(parts[1])
            except ValueError:
                pass

        schema[name] = {
            "module_type": module_type,
            "layer_idx": layer_idx,
        }

    return schema
