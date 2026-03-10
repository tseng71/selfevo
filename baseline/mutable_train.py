"""
SelfEvo - Mutable Training Script
This is the SOLE optimization target of the SelfEvo system.
A tiny decoder-only Transformer trained on TinyStories.

IMMUTABLE: validation logic, output schema, budget enforcement.
MUTABLE: model config, architecture details, optimizer, schedule, batching.
"""

import json
import math
import time
import os
import sys
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

# === CONFIG === (MUTABLE)
n_layer = 6
n_head = 2
n_embd = 128
block_size = 256
dropout = 0.02
bias = False

learning_rate = 1e-3
weight_decay = 0.01
beta1 = 0.9
beta2 = 0.999
grad_clip = 0.3

batch_size = 96
gradient_accumulation_steps = 2
warmup_steps = 150
max_steps = 500

eval_interval = 50
eval_steps = 20
# === END CONFIG ===

# === PATHS === (IMMUTABLE)
DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data")
TRAIN_BIN = os.path.join(DATA_DIR, "train.bin")
VAL_BIN = os.path.join(DATA_DIR, "val.bin")
META_PATH = os.path.join(DATA_DIR, "meta.json")
# === END PATHS ===


# === MODEL === (MUTABLE)
class RMSNorm(nn.Module):
    def __init__(self, dim, eps=1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x):
        norm = torch.rsqrt(x.float().pow(2).mean(-1, keepdim=True) + self.eps)
        return (x.float() * norm).type_as(x) * self.weight


class SwiGLU(nn.Module):
    def __init__(self, dim, hidden_dim=None, dropout_rate=0.0):
        super().__init__()
        hidden_dim = hidden_dim or int(dim * 8 / 3)
        # Round to nearest multiple of 8 for efficiency
        hidden_dim = ((hidden_dim + 7) // 8) * 8
        self.w1 = nn.Linear(dim, hidden_dim, bias=bias)
        self.w2 = nn.Linear(hidden_dim, dim, bias=bias)
        self.w2.SCALE_INIT = True
        self.w3 = nn.Linear(dim, hidden_dim, bias=bias)
        self.dropout = nn.Dropout(dropout_rate)

    def forward(self, x):
        return self.dropout(self.w2(F.silu(self.w1(x)) * self.w3(x)))


class CausalSelfAttention(nn.Module):
    def __init__(self, dim, n_heads, dropout_rate=0.0):
        super().__init__()
        assert dim % n_heads == 0
        self.n_heads = n_heads
        self.head_dim = dim // n_heads
        self.qkv = nn.Linear(dim, 3 * dim, bias=bias)
        self.out_proj = nn.Linear(dim, dim, bias=bias)
        self.out_proj.SCALE_INIT = True
        self.attn_dropout = nn.Dropout(dropout_rate)
        self.resid_dropout = nn.Dropout(dropout_rate)

    def forward(self, x):
        B, T, C = x.shape
        qkv = self.qkv(x)
        q, k, v = qkv.split(C, dim=2)
        q = q.view(B, T, self.n_heads, self.head_dim).transpose(1, 2)
        k = k.view(B, T, self.n_heads, self.head_dim).transpose(1, 2)
        v = v.view(B, T, self.n_heads, self.head_dim).transpose(1, 2)

        # Fast QK-Norm to stabilize attention
        q = F.normalize(q, dim=-1) * (self.head_dim ** 0.5)
        k = F.normalize(k, dim=-1) * (self.head_dim ** 0.5)

        # Use PyTorch scaled dot product attention (flash attention when available)
        y = F.scaled_dot_product_attention(
            q, k, v, is_causal=True, dropout_p=self.attn_dropout.p if self.training else 0.0
        )
        y = y.transpose(1, 2).contiguous().view(B, T, C)
        return self.resid_dropout(self.out_proj(y))


class TransformerBlock(nn.Module):
    def __init__(self, dim, n_heads, dropout_rate=0.0):
        super().__init__()
        self.ln1 = RMSNorm(dim)
        self.attn = CausalSelfAttention(dim, n_heads, dropout_rate)
        self.ln2 = RMSNorm(dim)
        self.mlp = SwiGLU(dim, dropout_rate=dropout_rate)

    def forward(self, x):
        x = x + self.attn(self.ln1(x))
        x = x + self.mlp(self.ln2(x))
        return x


class TinyTransformer(nn.Module):
    def __init__(self, vocab_size):
        super().__init__()
        self.token_emb = nn.Embedding(vocab_size, n_embd)
        self.drop = nn.Dropout(dropout)
        self.blocks = nn.ModuleList(
            [TransformerBlock(n_embd, n_head, dropout) for _ in range(n_layer)]
        )
        self.ln_f = RMSNorm(n_embd)
        self.lm_head = nn.Linear(n_embd, vocab_size, bias=False)
        # Weight tying
        self.token_emb.weight = self.lm_head.weight
        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            std = 0.02
            if hasattr(module, 'SCALE_INIT'):
                std *= (2 * n_layer) ** -0.5
            torch.nn.init.normal_(module.weight, mean=0.0, std=std)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(self, idx):
        B, T = idx.shape
        assert T <= block_size, f"Sequence length {T} exceeds block_size {block_size}"
        tok_emb = self.token_emb(idx)
        pos = torch.arange(T, device=idx.device).unsqueeze(1).float()
        div = torch.exp(torch.arange(0, n_embd, 2, device=idx.device).float() * (-math.log(10000.0) / n_embd))
        pos_emb = torch.zeros(T, n_embd, device=idx.device)
        pos_emb[:, 0::2] = torch.sin(pos * div)
        pos_emb[:, 1::2] = torch.cos(pos * div)
        x = self.drop(tok_emb + pos_emb * 0.02)
        for block in self.blocks:
            x = block(x)
        x = self.ln_f(x)
        logits = self.lm_head(x)
        return logits

    def count_params(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
# === END MODEL ===


# === DATA LOADING === (MUTABLE)
def load_data(split):
    path = TRAIN_BIN if split == "train" else VAL_BIN
    data = np.memmap(path, dtype=np.uint16, mode="r")
    return data


def get_batch(data, device):
    ix = torch.randint(len(data) - block_size - 1, (batch_size,))
    x = torch.stack([torch.from_numpy(data[i : i + block_size].astype(np.int64)) for i in ix])
    y = torch.stack([torch.from_numpy(data[i + 1 : i + 1 + block_size].astype(np.int64)) for i in ix])
    return x.to(device), y.to(device)
# === END DATA LOADING ===


# === TRAINING === (MUTABLE - optimizer, schedule)
def get_lr(step):
    """WSD learning rate schedule."""
    if step < warmup_steps:
        return learning_rate * (step + 1) / warmup_steps
    if step >= max_steps:
        return learning_rate * 0.1
    decay_steps = int(max_steps * 0.2)
    stable_steps = max_steps - warmup_steps - decay_steps
    if step < warmup_steps + stable_steps:
        return learning_rate
    decay_ratio = (step - warmup_steps - stable_steps) / decay_steps
    coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio))
    return learning_rate * 0.1 + coeff * (learning_rate - learning_rate * 0.1)
# === END TRAINING ===


# === EVALUATION === (IMMUTABLE)
@torch.no_grad()
def evaluate(model, val_data, device):
    model.eval()
    losses = []
    for _ in range(eval_steps):
        x, y = get_batch(val_data, device)
        logits = model(x)
        loss = F.cross_entropy(logits.view(-1, logits.size(-1)), y.view(-1))
        losses.append(loss.item())
    model.train()
    return float(np.mean(losses))
# === END EVALUATION ===


# === MAIN === (MUTABLE - except OUTPUT schema which is IMMUTABLE)
def main():
    start_time = time.time()

    # Load metadata
    with open(META_PATH) as f:
        meta = json.load(f)
    vocab_size = meta["vocab_size"]

    # Device selection
    if torch.backends.mps.is_available():
        device = torch.device("mps")
    elif torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")

    # Load data
    train_data = load_data("train")
    val_data = load_data("val")

    # Build model
    model = TinyTransformer(vocab_size).to(device)
    num_params = model.count_params()

    # Optimizer
    decay_params = [p for p in model.parameters() if p.dim() >= 2]
    nodecay_params = [p for p in model.parameters() if p.dim() < 2]
    optimizer = torch.optim.AdamW(
        [
            {"params": decay_params, "weight_decay": weight_decay},
            {"params": nodecay_params, "weight_decay": 0.0}
        ],
        lr=learning_rate,
        betas=(beta1, beta2),
    )

    # Training loop
    model.train()
    train_start = time.time()
    best_val_loss = float("inf")
    status = "ok"
    peak_mem_mb = 0.0

    try:
        for step in range(max_steps):
            # LR schedule
            lr = get_lr(step)
            for param_group in optimizer.param_groups:
                param_group["lr"] = lr

            # Gradient accumulation
            optimizer.zero_grad()
            accum_loss = 0.0
            for micro_step in range(gradient_accumulation_steps):
                x, y = get_batch(train_data, device)
                logits = model(x)
                loss = F.cross_entropy(logits.view(-1, logits.size(-1)), y.view(-1))
                loss = loss / gradient_accumulation_steps
                loss.backward()
                accum_loss += loss.item()

            # Gradient clipping
            if grad_clip > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)

            optimizer.step()

            # Track memory
            if device.type == "mps":
                mem = torch.mps.driver_allocated_memory() / (1024 * 1024)
            elif device.type == "cuda":
                mem = torch.cuda.max_memory_allocated() / (1024 * 1024)
            else:
                mem = 0.0
            peak_mem_mb = max(peak_mem_mb, mem)

            # Periodic evaluation
            if (step + 1) % eval_interval == 0 or step == max_steps - 1:
                val_loss = evaluate(model, val_data, device)
                best_val_loss = min(best_val_loss, val_loss)

                # NaN check
                if math.isnan(val_loss) or math.isinf(val_loss):
                    status = "nan"
                    break

    except RuntimeError as e:
        if "out of memory" in str(e).lower():
            status = "oom"
        else:
            status = "error"
        best_val_loss = float("inf")
    except Exception as e:
        status = "error"
        best_val_loss = float("inf")

    train_time = time.time() - train_start
    total_time = time.time() - start_time

    # Final evaluation if training completed normally
    if status == "ok":
        val_loss = evaluate(model, val_data, device)
        best_val_loss = min(best_val_loss, val_loss)

    # === OUTPUT === (IMMUTABLE SCHEMA)
    result = {
        "val_loss": round(best_val_loss, 6) if best_val_loss != float("inf") else None,
        "train_time_sec": round(train_time, 2),
        "total_time_sec": round(total_time, 2),
        "peak_mem_mb": round(peak_mem_mb, 1),
        "num_steps": max_steps,
        "num_params": num_params,
        "status": status,
    }
    print(json.dumps(result))
    return result


# === END MAIN ===


if __name__ == "__main__":
    main()
