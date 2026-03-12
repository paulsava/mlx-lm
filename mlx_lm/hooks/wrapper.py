import mlx.nn as nn

from .hooked_module import HookedRootModule
from .injectors import (
    inject_attention_hooks,
    inject_gated_attention_hooks,
    inject_block_hooks,
    inject_linear_attention_hooks,
    inject_mlp_hooks,
)


def _is_gated_attention(attn_module: nn.Module) -> bool:
    """Detect gated attention (e.g. Qwen3.5) where q_proj outputs 2x head_dim."""
    if not hasattr(attn_module, "q_proj") or not hasattr(attn_module, "head_dim"):
        return False
    n_heads = getattr(attn_module, "num_attention_heads", None) or getattr(attn_module, "n_heads", None)
    if n_heads is None:
        return False
    q_out_features = attn_module.q_proj.weight.shape[0]
    return q_out_features == n_heads * attn_module.head_dim * 2


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
    # or model.language_model.model.layers (VLM-style wrappers like Qwen3.5)
    layers = None
    if hasattr(model, "model") and hasattr(model.model, "layers"):
        layers = model.model.layers
    elif hasattr(model, "language_model") and hasattr(model.language_model, "model") and hasattr(model.language_model.model, "layers"):
        layers = model.language_model.model.layers

    if layers is None:
        raise ValueError(
            "Model does not have expected structure. "
            "Expected model.model.layers or model.language_model.model.layers. "
            "This hook system currently only supports standard MLX-LM models."
        )

    for i, layer in enumerate(layers):
        # Inject block-level residual hooks
        inject_block_hooks(layer, i, model)

        # Detect and inject attention hooks
        # Support both standard softmax attention and linear attention (GatedDeltaNet)
        if hasattr(layer, "linear_attn"):
            inject_linear_attention_hooks(layer.linear_attn, i, model)
        else:
            attn_module = None
            for attr_name in ["self_attn", "attn", "attention"]:
                if hasattr(layer, attr_name):
                    attn_module = getattr(layer, attr_name)
                    break

            if attn_module is not None:
                if _is_gated_attention(attn_module):
                    inject_gated_attention_hooks(attn_module, i, model, use_manual_sdpa)
                else:
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
