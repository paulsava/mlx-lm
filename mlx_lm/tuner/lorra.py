"""LoRRA: Low-Rank Representation Adaptation

Train LoRA adapters to bake steering vector effects into model weights.
Instead of standard language modeling loss, LoRRA uses a representation
matching loss: push the model's hidden states toward
(frozen_hidden + α * steering_direction).

Based on: "Representation Engineering" (Zou et al. 2023, arXiv:2310.01405)
"""

from __future__ import annotations

import json
import re
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import mlx.core as mx
import mlx.nn as nn
import mlx.optimizers as opt
import numpy as np
from mlx.utils import tree_flatten
from mlx_lm.hooks import wrap_model_with_hooks


# ── Steering vector loading ─────────────────────────────────────────


def parse_layer_idx(name: str) -> int:
    """Extract layer index from a hook name like 'blocks.35.hook_resid_post'."""
    match = re.search(r"blocks\.(\d+)\.", name)
    if match is None:
        raise ValueError(f"Could not parse layer from {name}")
    return int(match.group(1))


def load_steering_vectors(path: str | Path) -> dict[int, mx.array]:
    """Load steering vectors from a safetensors file.

    Returns dict mapping layer_index -> vector (shape: hidden_size).
    """
    data = mx.load(str(path))
    vectors: dict[int, mx.array] = {}
    for key in data.keys():
        if "hook_resid_post" in key:
            layer_idx = parse_layer_idx(key)
            v = data[key].astype(mx.float32)
            mx.eval(v)
            vectors[layer_idx] = v
    return vectors


# ── Hidden state collection ──────────────────────────────────────────


def collect_hidden_states(
    model: nn.Module,
    input_ids: mx.array,
    target_layers: list[int],
) -> dict[int, mx.array]:
    """Run forward pass, collect hidden states at target layers via hooks.

    Returns dict mapping layer_index -> tensor (batch, seq_len, hidden).
    """
    cache: dict[int, mx.array] = {}

    def make_collector(layer_idx: int):
        def hook_fn(name: str, activation: mx.array, context: dict) -> mx.array:
            cache[layer_idx] = activation
            return activation
        return hook_fn

    for L in target_layers:
        model.add_hook(f"blocks.{L}.hook_resid_post", make_collector(L))

    logits = model(input_ids)
    mx.eval(logits)
    model.reset_hooks()

    return cache


# ── LoRRA training config ────────────────────────────────────────────


@dataclass
class LorraConfig:
    """Configuration for LoRRA training."""

    # Steering
    steering_vectors_path: str = ""
    target_layers: list[int] = field(default_factory=lambda: [29, 31, 33, 35, 37, 39])
    alpha: float = 5.0

    # Training
    lr: float = 3e-4
    epochs: int = 3
    max_seq_length: int = 128
    report_every: int = 5
    save_every_epoch: bool = True

    # Tokenization
    enable_thinking: bool = False

    # Output
    output_dir: str = "outputs/lorra"


# ── Training loop ────────────────────────────────────────────────────


def train_lorra(
    model: nn.Module,
    tokenizer: Any,
    prompts: list[str],
    config: LorraConfig,
    hooked_model: nn.Module | None = None,
):
    """Run LoRRA training: bake steering vector effects into LoRA weights.

    This is a self-contained training loop that:
    1. Uses hooked model for activation collection
    2. Pre-computes frozen hidden states for all prompts
    3. Trains LoRA adapters with representation matching loss
    4. Saves the trained adapters

    Args:
        model: Model with LoRA adapters already applied (frozen base + unfrozen LoRA).
        hooked_model: Model wrapped with hooks (must be done before LoRA application).
        tokenizer: Tokenizer for encoding prompts.
        prompts: List of training prompt strings.
        config: LoRRA training configuration.
    """
    output_dir = Path(config.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    target_layers = config.target_layers

    # Load steering vectors
    print(f"Loading steering vectors from {config.steering_vectors_path}...")
    steer_vecs = load_steering_vectors(config.steering_vectors_path)

    # Verify vectors exist for all target layers
    for L in target_layers:
        if L not in steer_vecs:
            raise ValueError(f"No steering vector for target layer {L}")
        norm = float(mx.sqrt(mx.sum(steer_vecs[L] * steer_vecs[L])))
        print(f"  L{L}: norm={norm:.2f}")

    # Pre-compute target offsets: α * steering_vector
    target_offsets: dict[int, mx.array] = {}
    for L in target_layers:
        offset = config.alpha * steer_vecs[L]
        mx.eval(offset)
        target_offsets[L] = offset

    # Tokenize prompts
    print(f"\nTokenizing {len(prompts)} prompts...")
    tokenized: list[list[int]] = []
    for prompt in prompts:
        messages = [{"role": "user", "content": prompt}]
        kwargs = dict(return_dict=False, add_generation_prompt=True)
        if not config.enable_thinking:
            kwargs["enable_thinking"] = False
        tokens = tokenizer.apply_chat_template(messages, **kwargs)
        if len(tokens) <= config.max_seq_length:
            tokenized.append(tokens)
        else:
            tokenized.append(tokens[:config.max_seq_length])
    print(f"  {len(tokenized)} prompts tokenized")

    # Use pre-wrapped hooked model, or wrap now if not provided
    if hooked_model is not None:
        hooked = hooked_model
    else:
        hooked = wrap_model_with_hooks(model)

    # ── Pre-compute frozen hidden states ─────────────────────────────
    # Zero out LoRA params temporarily to get true baseline activations
    model.eval()  # Use fast Metal kernels for inference-only collection
    print("\nPre-computing frozen hidden states...")
    lora_state = dict(tree_flatten(model.trainable_parameters()))
    zero_state = {k: mx.zeros_like(v) for k, v in lora_state.items()}
    model.load_weights(list(zero_state.items()), strict=False)
    mx.eval(model.parameters())

    frozen_hiddens: list[dict[int, mx.array]] = []
    for idx, tokens in enumerate(tokenized):
        input_ids = mx.array(tokens[:-1]).reshape(1, -1)  # inputs = all but last
        hiddens = collect_hidden_states(hooked, input_ids, target_layers)
        hiddens = {L: mx.array(h) for L, h in hiddens.items()}
        for h in hiddens.values():
            mx.eval(h)
        frozen_hiddens.append(hiddens)
        if (idx + 1) % 10 == 0:
            print(f"  {idx+1}/{len(tokenized)}")
        mx.clear_cache()
    print(f"  Cached {len(frozen_hiddens)} frozen hidden states")

    # Restore LoRA params (still zeros since we just started)
    model.load_weights(list(lora_state.items()), strict=False)
    mx.eval(model.parameters())

    # ── Loss function ────────────────────────────────────────────────

    def loss_fn(model_for_grad, input_ids, frozen_h):
        """LoRRA loss for a single example.

        Loss = mean over target layers of per-position L2 distance
        between LoRA-adapted hidden states and target hidden states.
        """
        cache: dict[int, mx.array] = {}

        def make_collector(layer_idx):
            def hook_fn(name, activation, context):
                cache[layer_idx] = activation
                return activation
            return hook_fn

        for L in target_layers:
            hooked.add_hook(f"blocks.{L}.hook_resid_post", make_collector(L))

        _logits = hooked(input_ids)
        hooked.reset_hooks()

        total_loss = mx.array(0.0)
        n = 0
        for L in target_layers:
            if L not in cache or L not in frozen_h:
                continue

            lora_h = cache[L]           # (1, seq_len, hidden)
            fh = frozen_h[L]            # (1, seq_len, hidden)
            seq_len = lora_h.shape[1]
            fh = fh[:, :seq_len, :]     # truncate if needed
            target = fh + target_offsets[L]  # broadcast (hidden,)

            diff = lora_h - target
            l2 = mx.sqrt(mx.sum(diff * diff, axis=-1) + 1e-8)  # (1, seq_len)
            total_loss = total_loss + mx.mean(l2)
            n += 1

        if n > 0:
            total_loss = total_loss / n
        return total_loss

    loss_and_grad = nn.value_and_grad(model, loss_fn)

    # ── Training ─────────────────────────────────────────────────────
    print(f"\nStarting LoRRA training...")
    print(f"  Epochs: {config.epochs}")
    print(f"  Prompts: {len(tokenized)}")
    print(f"  Target layers: {target_layers}")
    print(f"  Alpha: {config.alpha}")
    print(f"  LR: {config.lr}")

    model.train()  # Enable training mode (uses ops fallback for GatedDeltaNet VJP)
    optimizer = opt.Adam(learning_rate=config.lr)
    global_step = 0

    for epoch in range(config.epochs):
        print(f"\n  Epoch {epoch+1}/{config.epochs}")
        indices = list(range(len(tokenized)))
        np.random.shuffle(indices)

        epoch_loss = 0.0
        epoch_steps = 0
        t_epoch = time.time()

        for i, idx in enumerate(indices):
            tokens = tokenized[idx]
            input_ids = mx.array(tokens[:-1]).reshape(1, -1)
            frozen_h = frozen_hiddens[idx]

            t0 = time.time()
            loss_val, grads = loss_and_grad(model, input_ids, frozen_h)
            optimizer.update(model, grads)
            mx.eval(model.parameters(), optimizer.state)

            loss_f = float(loss_val)
            epoch_loss += loss_f
            epoch_steps += 1
            global_step += 1

            if global_step % config.report_every == 0:
                avg = epoch_loss / epoch_steps
                dt = time.time() - t0
                peak_mem = mx.get_peak_memory() / 1e9
                print(
                    f"    Step {global_step:4d} | loss={loss_f:.4f} | "
                    f"avg={avg:.4f} | {dt:.1f}s/step | "
                    f"peak_mem={peak_mem:.1f}GB"
                )

            mx.clear_cache()

        avg_loss = epoch_loss / max(epoch_steps, 1)
        elapsed = time.time() - t_epoch
        print(f"  Epoch {epoch+1}: avg_loss={avg_loss:.4f} ({elapsed:.0f}s)")

        if config.save_every_epoch:
            adapter_weights = dict(tree_flatten(model.trainable_parameters()))
            path = output_dir / f"adapters_epoch{epoch+1}.safetensors"
            mx.save_safetensors(str(path), adapter_weights)
            print(f"  Saved: {path}")

    # Save final adapters
    adapter_weights = dict(tree_flatten(model.trainable_parameters()))
    final_path = output_dir / "adapters.safetensors"
    mx.save_safetensors(str(final_path), adapter_weights)
    print(f"\nFinal adapters saved to {final_path}")

    # Save config
    config_dict = {
        "target_layers": target_layers,
        "alpha": config.alpha,
        "lr": config.lr,
        "epochs": config.epochs,
        "num_prompts": len(tokenized),
        "steering_vectors_path": str(config.steering_vectors_path),
    }
    config_path = output_dir / "lorra_config.json"
    with config_path.open("w") as f:
        json.dump(config_dict, f, indent=2)
    print(f"Config saved to {config_path}")

    return final_path
