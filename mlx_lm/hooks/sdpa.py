from typing import Callable, Optional, Union

import mlx.core as mx


def _maybe_repeat_kv(tensor: mx.array, repeats: int) -> mx.array:
    """Repeat KV heads to match the number of query heads."""
    if repeats == 1:
        return tensor
    return mx.repeat(tensor, repeats, axis=1)


def _apply_attention_mask(
    attn_scores: mx.array,
    mask: Union[str, mx.array],
) -> mx.array:
    """Apply attention mask following mx.fast.scaled_dot_product_attention semantics."""
    if isinstance(mask, str):
        if mask != "causal":
            raise ValueError(f"Unsupported attention mask: {mask}")
        q_len, k_len = attn_scores.shape[-2:]
        # Match MLX causal masking logic (see quantized_scaled_dot_product_attention)
        q_indices = mx.arange(k_len - q_len, k_len)
        k_indices = mx.arange(k_len)
        mask = q_indices[:, None] >= k_indices[None]

    if not isinstance(mask, mx.array):
        mask = mx.array(mask)

    if mask.dtype == mx.bool_:
        min_val = mx.finfo(attn_scores.dtype).min
        attn_scores = mx.where(mask, attn_scores, min_val)
    else:
        attn_scores = attn_scores + mask
    return attn_scores


def manual_scaled_dot_product_attention(
    queries: mx.array,
    keys: mx.array,
    values: mx.array,
    scale: float,
    mask: Optional[Union[str, mx.array]] = None,
    attn_pattern_hook: Optional[Callable[[mx.array], mx.array]] = None,
) -> tuple[mx.array, Optional[mx.array]]:
    """Compute SDPA with manual control while matching MX's fused-kernel semantics."""
    B, n_q_heads, q_len, _ = queries.shape
    _, n_kv_heads, k_len, _ = keys.shape

    if n_q_heads % n_kv_heads != 0:
        raise ValueError(
            f"Query heads ({n_q_heads}) must be divisible by key/value heads ({n_kv_heads})."
        )

    repeats = n_q_heads // n_kv_heads
    keys = _maybe_repeat_kv(keys, repeats)
    values = _maybe_repeat_kv(values, repeats)

    # Compute attention scores (B, n_q_heads, L_q, L_k)
    attn_scores = (queries * scale) @ keys.transpose(0, 1, 3, 2)

    if mask is not None:
        attn_scores = _apply_attention_mask(attn_scores, mask)

    attn_weights = mx.softmax(attn_scores, axis=-1, precise=True)

    attn_pattern = None
    if attn_pattern_hook is not None:
        attn_weights = attn_pattern_hook(attn_weights)
        attn_pattern = attn_weights

    output = attn_weights @ values

    return output, attn_pattern
