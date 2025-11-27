"""
Basic usage example for MLX-LM hooks.

Demonstrates:
1. Loading a model and wrapping with hooks
2. Listing available hook points
3. Caching activations with run_with_cache()
4. Modifying activations with custom hooks
"""

import mlx.core as mx

from mlx_lm import load
from mlx_lm.hooks import wrap_model_with_hooks


def main():
    # Load model (using a small model for demo)
    print("Loading model...")
    model, tokenizer = load("mlx-community/Llama-3.2-1B-Instruct-4bit")

    # Wrap with hooks
    print("\nWrapping model with hooks...")
    hooked_model = wrap_model_with_hooks(model, use_manual_sdpa=True)

    # List available hook points
    print(f"\nAvailable hook points: {len(hooked_model.hook_points())}")
    print("\nSample hook points:")
    for hook_name in sorted(hooked_model.hook_points())[:10]:
        print(f"  - {hook_name}")

    # Prepare input
    prompt = "The capital of France is"
    tokens = tokenizer.encode(prompt)
    input_ids = mx.array([tokens])

    # Run with cache to capture all activations
    print(f"\nRunning model with cache on input: '{prompt}'")
    output, cache = hooked_model.run_with_cache(input_ids)

    # Inspect cached activations
    print(f"\nCached activations: {len(cache)} hook points")
    print("\nSample activation shapes:")
    for hook_name in sorted(cache.keys())[:5]:
        print(f"  {hook_name}: {cache[hook_name].shape}")

    # Example: Access attention patterns (if manual SDPA enabled)
    if "blocks.0.attn.hook_attn_pattern" in cache:
        attn_patterns = cache["blocks.0.attn.hook_attn_pattern"]
        print(f"\nAttention patterns shape: {attn_patterns.shape}")
        print("  (batch_size, num_heads, seq_len, seq_len)")

    # Example: Modify activations with a hook
    print("\n\nExample: Zeroing out attention head 0 in layer 5")

    def zero_head_0(name, activation, context):
        """Zero out the first attention head."""
        # Copy for MLX immutability
        activation = mx.array(activation)
        activation[:, 0, :, :] = 0
        return activation

    # Add hook
    hooked_model.add_hook("blocks.5.attn.hook_q_rotated", zero_head_0)

    # Run model with modified activations
    output_modified = hooked_model(input_ids)

    # Clean up hooks
    hooked_model.reset_hooks()

    print("\nModel ran successfully with modified activations!")
    print("Output shape:", output_modified.shape)


if __name__ == "__main__":
    main()
