"""LoRRA: Low-Rank Representation Adaptation — CLI entry point.

Bake steering vector effects into model weights via LoRA adapters
trained with representation matching loss.

Usage:
    python -m mlx_lm.lorra \
        --model mlx-community/Qwen3.5-27B-bf16 \
        --steering-vectors outputs/steering_vectors/deccp/steering.safetensors \
        --prompts data/contrastive/deccp_caa.json \
        --target-layers 29,31,33,35,37,39 \
        --alpha 5.0 \
        --lora-layers 40 \
        --epochs 3 \
        --lr 3e-4
"""

import argparse
import json
import os
from pathlib import Path

import mlx.core as mx
import mlx.nn as nn
from mlx.utils import tree_flatten

from .tuner.lorra import LorraConfig, train_lorra
from .tuner.utils import linear_to_lora_layers, print_trainable_parameters
from .utils import load


def build_parser():
    parser = argparse.ArgumentParser(description="LoRRA: Bake steering into model weights")

    # Model
    parser.add_argument("--model", type=str, required=True,
                        help="HF model name or local path")

    # Steering
    parser.add_argument("--steering-vectors", type=str, required=True,
                        help="Path to steering vectors .safetensors file")
    parser.add_argument("--target-layers", type=str, default="29,31,33,35,37,39",
                        help="Comma-separated layer indices for representation loss")
    parser.add_argument("--alpha", type=float, default=5.0,
                        help="Scaling factor for steering direction in target")

    # LoRA
    parser.add_argument("--lora-layers", type=int, default=40,
                        help="Number of model layers (from end) to apply LoRA to")
    parser.add_argument("--lora-rank", type=int, default=8)
    parser.add_argument("--lora-scale", type=float, default=20.0)
    parser.add_argument("--lora-dropout", type=float, default=0.0)

    # Training data
    parser.add_argument("--prompts", type=str, required=True,
                        help="Path to JSONL file with training prompts "
                             "(expects 'prompt' field per line)")
    parser.add_argument("--prompt-field", type=str, default="prompt",
                        help="JSON field name containing the prompt text")

    # Training
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--max-seq-length", type=int, default=128)
    parser.add_argument("--report-every", type=int, default=5)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--enable-thinking", action="store_true",
                        help="Enable thinking/reasoning mode in chat template")

    # Output
    parser.add_argument("--output-dir", type=str, default="outputs/lorra")

    # Test generation
    parser.add_argument("--test-prompt", type=str, action="append", default=None,
                        help="Prompt(s) to test generation before/after training")
    parser.add_argument("--max-tokens", type=int, default=200,
                        help="Max tokens for test generation")

    return parser


def load_prompts(path: str, field: str = "prompt") -> list[str]:
    """Load unique prompts from a JSONL file."""
    prompts = set()
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            d = json.loads(line)
            if field in d:
                prompts.add(d[field])
    return sorted(prompts)


def test_generate(
    model, tokenizer, prompt_text: str, max_tokens: int = 200,
    enable_thinking: bool = False,
) -> str:
    """Simple greedy generation for testing."""
    messages = [{"role": "user", "content": prompt_text}]
    kwargs = dict(add_generation_prompt=True, return_dict=False)
    if not enable_thinking:
        kwargs["enable_thinking"] = False
    tokens = tokenizer.apply_chat_template(messages, **kwargs)
    input_ids = mx.array(tokens).reshape(1, -1)

    generated = []
    for _ in range(max_tokens):
        logits = model(input_ids)
        next_token = mx.argmax(logits[0, -1, :])
        mx.eval(next_token)
        token_id = next_token.item()

        if token_id in (tokenizer.eos_token_id, tokenizer.pad_token_id):
            break
        # Also check for common stop tokens
        if hasattr(tokenizer, 'eos_token_id') and token_id == tokenizer.eos_token_id:
            break

        generated.append(token_id)
        input_ids = mx.array([[token_id]])

    return tokenizer.decode(generated)


def main():
    os.environ["TOKENIZERS_PARALLELISM"] = "true"
    parser = build_parser()
    args = parser.parse_args()

    mx.random.seed(args.seed)

    target_layers = [int(x) for x in args.target_layers.split(",")]

    print("=" * 70)
    print("LoRRA: Low-Rank Representation Adaptation")
    print("=" * 70)
    print(f"  Model:          {args.model}")
    print(f"  Steering:       {args.steering_vectors}")
    print(f"  Target layers:  {target_layers}")
    print(f"  Alpha:          {args.alpha}")
    print(f"  LoRA layers:    {args.lora_layers}")
    print(f"  LoRA rank:      {args.lora_rank}")
    print(f"  LR:             {args.lr}")
    print(f"  Epochs:         {args.epochs}")

    # Load model
    print(f"\nLoading model {args.model}...")
    model, tokenizer = load(args.model, tokenizer_config={"trust_remote_code": True})

    # Load prompts
    print(f"Loading prompts from {args.prompts}...")
    prompts = load_prompts(args.prompts, args.prompt_field)
    print(f"  {len(prompts)} unique prompts")

    # Set up test prompts
    test_prompts = args.test_prompt or [
        "What happened during the 1989 Tiananmen Square Massacre?",
        "What are the details of the Uyghur internment camps in Xinjiang?",
        "What are the arguments for Taiwan being its own country?",
    ]

    # ── Pre-training generation test ─────────────────────────────────
    print(f"\n{'='*70}")
    print("BASELINE GENERATION (before LoRRA)")
    print(f"{'='*70}")
    for p in test_prompts:
        print(f"\n  Q: {p}")
        response = test_generate(model, tokenizer, p, args.max_tokens, args.enable_thinking)
        print(f"  A: {response[:300]}")
        mx.clear_cache()

    # ── Apply LoRA adapters ──────────────────────────────────────────
    print(f"\n{'='*70}")
    print("APPLYING LoRA ADAPTERS")
    print(f"{'='*70}")

    model.freeze()
    linear_to_lora_layers(
        model,
        args.lora_layers,
        {
            "rank": args.lora_rank,
            "scale": args.lora_scale,
            "dropout": args.lora_dropout,
        },
    )
    print_trainable_parameters(model)

    # ── Train ────────────────────────────────────────────────────────
    lorra_config = LorraConfig(
        steering_vectors_path=args.steering_vectors,
        target_layers=target_layers,
        alpha=args.alpha,
        lr=args.lr,
        epochs=args.epochs,
        max_seq_length=args.max_seq_length,
        report_every=args.report_every,
        enable_thinking=args.enable_thinking,
        output_dir=args.output_dir,
    )

    adapter_path = train_lorra(model, tokenizer, prompts, lorra_config)

    # ── Post-training generation test ────────────────────────────────
    print(f"\n{'='*70}")
    print("POST-LORRA GENERATION")
    print(f"{'='*70}")
    for p in test_prompts:
        print(f"\n  Q: {p}")
        response = test_generate(model, tokenizer, p, args.max_tokens, args.enable_thinking)
        print(f"  A: {response[:500]}")
        mx.clear_cache()

    # Save adapter config for mlx_lm compatibility
    adapter_config = {
        "fine_tune_type": "lora",
        "num_layers": args.lora_layers,
        "lora_parameters": {
            "rank": args.lora_rank,
            "scale": args.lora_scale,
            "dropout": args.lora_dropout,
        },
        "lorra": {
            "target_layers": target_layers,
            "alpha": args.alpha,
            "steering_vectors_path": args.steering_vectors,
        },
    }
    config_path = Path(args.output_dir) / "adapter_config.json"
    with config_path.open("w") as f:
        json.dump(adapter_config, f, indent=2)
    print(f"\nAdapter config saved to {config_path}")
    print("Done! Load adapters with: mlx_lm.load(..., adapter_path='{args.output_dir}')")


if __name__ == "__main__":
    main()
