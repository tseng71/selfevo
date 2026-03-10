"""
SelfEvo - Text Generation Script
Load the current best model and generate text to see its capabilities.
"""

import json
import sys
import os
import torch
import torch.nn.functional as F
from pathlib import Path

PROJECT_DIR = Path(__file__).parent
BASELINE_SCRIPT = PROJECT_DIR / "baseline" / "mutable_train.py"
DATA_DIR = PROJECT_DIR / "data"
META_PATH = DATA_DIR / "meta.json"
TOKENIZER_PATH = DATA_DIR / "tokenizer.json"


def load_model_from_baseline():
    """Dynamically load the model by executing baseline's mutable_train.py config + model."""
    import importlib.util

    # Load metadata
    with open(META_PATH) as f:
        meta = json.load(f)
    vocab_size = meta["vocab_size"]

    # We need to extract config and model from the baseline script
    # Execute it in a controlled way to get the model class
    baseline_code = BASELINE_SCRIPT.read_text()

    # Create a temporary module
    spec = importlib.util.spec_from_file_location("baseline_train", str(BASELINE_SCRIPT))
    mod = importlib.util.module_from_spec(spec)

    # We don't want it to run main(), just define the classes
    # Patch __name__ so it doesn't execute main
    mod.__name__ = "baseline_train"

    # Execute the module to define classes and config
    exec(compile(baseline_code, str(BASELINE_SCRIPT), "exec"), mod.__dict__)

    # Build model using the module's config
    model = mod.TinyTransformer(vocab_size)

    return model, mod


def load_tokenizer():
    from tokenizers import Tokenizer
    return Tokenizer.from_file(str(TOKENIZER_PATH))


@torch.no_grad()
def generate(model, tokenizer, prompt, max_new_tokens=200, temperature=0.8, top_k=40, device="cpu"):
    """Generate text from a prompt."""
    model.eval()

    # Encode prompt
    encoded = tokenizer.encode(prompt)
    ids = encoded.ids

    # Get block_size from model
    block_size = model.pos_emb.weight.shape[0]

    x = torch.tensor([ids], dtype=torch.long, device=device)

    generated_ids = list(ids)

    for _ in range(max_new_tokens):
        # Crop to block_size
        x_cond = x if x.shape[1] <= block_size else x[:, -block_size:]

        logits = model(x_cond)
        logits = logits[:, -1, :]  # last token

        # Temperature
        if temperature > 0:
            logits = logits / temperature

            # Top-k filtering
            if top_k > 0:
                v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                logits[logits < v[:, [-1]]] = -float("inf")

            probs = F.softmax(logits, dim=-1)
            next_id = torch.multinomial(probs, num_samples=1)
        else:
            next_id = logits.argmax(dim=-1, keepdim=True)

        generated_ids.append(next_id.item())
        x = torch.cat([x, next_id], dim=1)

        # Stop at end-of-text
        eot_id = tokenizer.token_to_id("<|endoftext|>")
        if next_id.item() == eot_id:
            break

    # Decode
    text = tokenizer.decode(generated_ids)
    return text


def train_and_save_checkpoint(model, mod, device, save_path):
    """Train the model briefly and save a checkpoint for generation."""
    import numpy as np

    # Load data
    train_data = np.memmap(str(DATA_DIR / "train.bin"), dtype=np.uint16, mode="r")
    val_data = np.memmap(str(DATA_DIR / "val.bin"), dtype=np.uint16, mode="r")

    model = model.to(device)

    # Get config from module
    lr = float(getattr(mod, 'learning_rate', 3e-4))
    wd = float(getattr(mod, 'weight_decay', 0.1))
    b1 = float(getattr(mod, 'beta1', 0.9))
    b2 = float(getattr(mod, 'beta2', 0.95))
    block_size = int(getattr(mod, 'block_size', 256))
    batch_size = int(getattr(mod, 'batch_size', 32))
    max_steps = int(getattr(mod, 'max_steps', 500))
    grad_clip = float(getattr(mod, 'grad_clip', 1.0))
    warmup_steps = int(getattr(mod, 'warmup_steps', 50))

    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, betas=(b1, b2), weight_decay=wd)

    model.train()
    import math

    print(f"Training {model.count_params():,} params for {max_steps} steps...")
    for step in range(max_steps):
        # LR schedule
        if step < warmup_steps:
            cur_lr = lr * (step + 1) / warmup_steps
        else:
            decay_ratio = (step - warmup_steps) / (max_steps - warmup_steps)
            cur_lr = lr * 0.1 + 0.5 * (1.0 + math.cos(math.pi * decay_ratio)) * (lr - lr * 0.1)
        for pg in optimizer.param_groups:
            pg["lr"] = cur_lr

        ix = torch.randint(len(train_data) - block_size - 1, (batch_size,))
        x = torch.stack([torch.from_numpy(train_data[i:i+block_size].astype(int)) for i in ix]).long().to(device)
        y = torch.stack([torch.from_numpy(train_data[i+1:i+1+block_size].astype(int)) for i in ix]).long().to(device)

        logits = model(x)
        loss = F.cross_entropy(logits.view(-1, logits.size(-1)), y.view(-1))

        optimizer.zero_grad()
        loss.backward()
        if grad_clip > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
        optimizer.step()

        if (step + 1) % 100 == 0:
            print(f"  step {step+1}/{max_steps}, loss={loss.item():.4f}")

    # Save checkpoint
    torch.save(model.state_dict(), save_path)
    print(f"Checkpoint saved to {save_path}")
    return model


def main():
    import argparse
    parser = argparse.ArgumentParser(description="Generate text with the current best model")
    parser.add_argument("--prompt", type=str, default="Once upon a time",
                        help="Text prompt to continue")
    parser.add_argument("--tokens", type=int, default=200,
                        help="Max tokens to generate")
    parser.add_argument("--temperature", type=float, default=0.8,
                        help="Sampling temperature (0=greedy)")
    parser.add_argument("--top-k", type=int, default=40,
                        help="Top-k sampling")
    parser.add_argument("--num", type=int, default=3,
                        help="Number of samples to generate")
    args = parser.parse_args()

    # Device
    if torch.backends.mps.is_available():
        device = torch.device("mps")
    elif torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")

    print(f"Device: {device}")

    # Load model definition from baseline
    print("Loading model from baseline config...")
    model, mod = load_model_from_baseline()
    print(f"Model: {model.count_params():,} parameters")

    # Check for existing checkpoint
    ckpt_path = PROJECT_DIR / "checkpoint.pt"
    if ckpt_path.exists():
        print(f"Loading checkpoint from {ckpt_path}...")
        state = torch.load(ckpt_path, map_location="cpu", weights_only=True)
        model.load_state_dict(state)
        model = model.to(device)
    else:
        print("No checkpoint found. Training from scratch...")
        model = train_and_save_checkpoint(model, mod, device, ckpt_path)

    # Load tokenizer
    tokenizer = load_tokenizer()

    # Generate
    print(f"\n{'='*60}")
    print(f"Prompt: \"{args.prompt}\"")
    print(f"Temperature: {args.temperature}, Top-k: {args.top_k}")
    print(f"{'='*60}\n")

    for i in range(args.num):
        print(f"--- Sample {i+1} ---")
        text = generate(model, tokenizer, args.prompt,
                       max_new_tokens=args.tokens,
                       temperature=args.temperature,
                       top_k=args.top_k,
                       device=device)
        print(text)
        print()


if __name__ == "__main__":
    main()
