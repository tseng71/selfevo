"""
SelfEvo - Data Preparation Module
Downloads TinyStories, trains a BPE tokenizer, tokenizes data, and caches as binary files.
"""

import os
import json
import struct
import numpy as np
from pathlib import Path

DATA_DIR = Path(__file__).parent / "data"
TOKENIZER_PATH = DATA_DIR / "tokenizer.json"
TRAIN_BIN = DATA_DIR / "train.bin"
VAL_BIN = DATA_DIR / "val.bin"
META_PATH = DATA_DIR / "meta.json"

VOCAB_SIZE = 4096
CONTEXT_LENGTH = 256
VAL_RATIO = 0.05
SEED = 42


def download_dataset():
    """Download TinyStories from HuggingFace."""
    from datasets import load_dataset
    print("[prepare] Downloading TinyStories dataset...")
    ds = load_dataset("roneneldan/TinyStories", split="train")
    print(f"[prepare] Downloaded {len(ds)} stories.")
    return ds


def train_tokenizer(texts):
    """Train a BPE tokenizer on the given texts."""
    from tokenizers import Tokenizer, models, trainers, pre_tokenizers, decoders

    print(f"[prepare] Training BPE tokenizer (vocab_size={VOCAB_SIZE})...")
    tokenizer = Tokenizer(models.BPE())
    tokenizer.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
    tokenizer.decoder = decoders.ByteLevel()

    trainer = trainers.BpeTrainer(
        vocab_size=VOCAB_SIZE,
        special_tokens=["<|endoftext|>", "<|padding|>"],
        show_progress=True,
    )

    # Train from iterator
    batch_size = 1000
    def batch_iterator():
        for i in range(0, len(texts), batch_size):
            yield texts[i : i + batch_size]

    tokenizer.train_from_iterator(batch_iterator(), trainer=trainer, length=len(texts))
    tokenizer.save(str(TOKENIZER_PATH))
    print(f"[prepare] Tokenizer saved to {TOKENIZER_PATH}")
    return tokenizer


def load_tokenizer():
    """Load a previously trained tokenizer."""
    from tokenizers import Tokenizer
    return Tokenizer.from_file(str(TOKENIZER_PATH))


def tokenize_and_save(texts, tokenizer):
    """Tokenize all texts and save as binary train/val splits."""
    print("[prepare] Tokenizing all stories...")
    eot_id = tokenizer.token_to_id("<|endoftext|>")

    all_ids = []
    for i, text in enumerate(texts):
        encoded = tokenizer.encode(text)
        all_ids.extend(encoded.ids)
        all_ids.append(eot_id)
        if (i + 1) % 50000 == 0:
            print(f"  ... tokenized {i + 1}/{len(texts)} stories")

    all_ids = np.array(all_ids, dtype=np.uint16)
    total = len(all_ids)
    print(f"[prepare] Total tokens: {total:,}")

    # Deterministic split
    rng = np.random.RandomState(SEED)
    indices = rng.permutation(total)
    val_size = int(total * VAL_RATIO)
    train_size = total - val_size

    # For simplicity, split contiguously (stories are already shuffled)
    train_ids = all_ids[:train_size]
    val_ids = all_ids[train_size:]

    train_ids.tofile(str(TRAIN_BIN))
    val_ids.tofile(str(VAL_BIN))
    print(f"[prepare] train.bin: {train_size:,} tokens ({TRAIN_BIN})")
    print(f"[prepare] val.bin:   {val_size:,} tokens ({VAL_BIN})")

    return train_size, val_size


def save_meta(vocab_size, train_tokens, val_tokens):
    """Save metadata about the prepared dataset."""
    meta = {
        "vocab_size": vocab_size,
        "train_tokens": train_tokens,
        "val_tokens": val_tokens,
        "context_length": CONTEXT_LENGTH,
        "tokenizer_path": str(TOKENIZER_PATH),
        "val_ratio": VAL_RATIO,
        "seed": SEED,
    }
    with open(META_PATH, "w") as f:
        json.dump(meta, f, indent=2)
    print(f"[prepare] Metadata saved to {META_PATH}")
    return meta


def main():
    DATA_DIR.mkdir(parents=True, exist_ok=True)

    # Check if already prepared
    if TRAIN_BIN.exists() and VAL_BIN.exists() and TOKENIZER_PATH.exists() and META_PATH.exists():
        meta = json.loads(META_PATH.read_text())
        print("[prepare] Data already prepared. Skipping.")
        print(f"  vocab_size:     {meta['vocab_size']}")
        print(f"  train_tokens:   {meta['train_tokens']:,}")
        print(f"  val_tokens:     {meta['val_tokens']:,}")
        print(f"  context_length: {meta['context_length']}")
        return meta

    # Step 1: Download
    ds = download_dataset()
    texts = ds["text"]

    # Step 2: Train or load tokenizer
    if TOKENIZER_PATH.exists():
        print("[prepare] Loading existing tokenizer...")
        tokenizer = load_tokenizer()
    else:
        tokenizer = train_tokenizer(texts)

    # Step 3: Tokenize and save
    train_tokens, val_tokens = tokenize_and_save(texts, tokenizer)

    # Step 4: Save metadata
    meta = save_meta(VOCAB_SIZE, train_tokens, val_tokens)

    print("\n[prepare] === Summary ===")
    print(f"  vocab_size:     {VOCAB_SIZE}")
    print(f"  train_tokens:   {train_tokens:,}")
    print(f"  val_tokens:     {val_tokens:,}")
    print(f"  context_length: {CONTEXT_LENGTH}")
    print("[prepare] Done!")
    return meta


if __name__ == "__main__":
    main()
