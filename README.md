# SelfEvo - Self-Evolving AI Training System

SelfEvo is a local self-evolving experiment system inspired by Andrej Karpathy's [autoresearch](https://github.com/karpathy/autoresearch) concept. It automatically and iteratively improves a tiny language model training script through an autonomous loop of code modification, training, evaluation, and selection.

## What It Does

SelfEvo wraps a tiny decoder-only Transformer (trained on [TinyStories](https://huggingface.co/datasets/roneneldan/TinyStories)) in a closed-loop evolution cycle:

1. **Propose** — An AI policy (Gemini / OpenAI / Claude) analyzes experiment history and proposes code modifications to the training script
2. **Train** — The modified script runs a fixed-budget training session
3. **Judge** — Results are compared against the current best (baseline) using `val_loss`
4. **Keep or Discard** — Better results update the baseline; worse results are rolled back
5. **Learn** — All experiments are logged, informing future proposals

A local web dashboard lets you observe progress, review experiments, and intervene — no programming knowledge required.

## Key Features

- **Single optimization target**: Only `mutable_train.py` is modified — evaluation logic is frozen
- **AI-powered policy**: Supports Gemini, OpenAI, and Claude for intelligent code modifications (falls back to heuristic rules if no API key is set)
- **Automatic rollback**: Failed or regressed experiments are safely discarded
- **Full experiment history**: Every round is logged in `memory.jsonl`
- **Local web dashboard**: Visual overview of progress, trends, and controls
- **Human-in-the-loop**: Pause, resume, approve high-risk experiments, or rollback manually

## Runs on Personal Computers

**No NVIDIA GPU required.** SelfEvo is designed to run on everyday personal computers:

| Platform | GPU Support | Status |
|----------|------------|--------|
| **macOS** (Apple Silicon iMac/MacBook) | MPS acceleration | Fully supported |
| **Windows** | NVIDIA CUDA / CPU | Supported |
| **Linux** | NVIDIA CUDA / CPU | Supported |

The system automatically detects available hardware and uses the best backend (MPS → CUDA → CPU). The tiny model size means even CPU-only training completes in minutes.

## Quick Start

### 1. Clone and install dependencies

```bash
git clone https://github.com/tseng71/selfevo.git
cd selfevo
pip install -r requirements.txt
```

### 2. Set up your AI API key

The AI policy module supports **three providers** — set any one:

| Provider | Env Variable | Get a Key |
|----------|-------------|-----------|
| Google Gemini | `GEMINI_API_KEY` | [aistudio.google.com](https://aistudio.google.com/apikey) |
| OpenAI (GPT-4o / GPT-5.4) | `OPENAI_API_KEY` | [platform.openai.com](https://platform.openai.com/api-keys) |
| Anthropic Claude (Opus 4.6) | `ANTHROPIC_API_KEY` | [console.anthropic.com](https://console.anthropic.com/) |

```bash
# Pick one:
export GEMINI_API_KEY="your-key"
export OPENAI_API_KEY="your-key"
export ANTHROPIC_API_KEY="your-key"
```

If multiple keys are set, priority is Gemini → OpenAI → Claude. Override with `AI_PROVIDER=openai` or `AI_PROVIDER=claude`. Override the model with `AI_MODEL=gpt-5.4` etc.

Without any API key, the system falls back to built-in heuristic mutations.

### 3. Prepare data

```bash
python prepare.py
```

This downloads TinyStories, trains a BPE tokenizer, and creates cached binary files.

### 4. Run

```bash
python main.py
```

Then open the dashboard URL shown in the terminal (default: http://localhost:8000).

The experiment loop starts automatically. Use the controls on the dashboard **Overview** page to:

- **Pause** — stop the loop after the current experiment finishes
- **Resume** — continue the loop
- **Single Step** — run exactly one experiment, then pause

You can also run with flags:

```bash
python main.py --no-dashboard    # Run experiment loop only (no web UI)
python main.py --dashboard-only  # Start dashboard without experiment loop
python main.py --single-step     # Run one experiment and exit
```

## Project Structure

```
selfevo/
├── main.py               # Entry point: dashboard + experiment loop
├── prepare.py            # Data preparation (TinyStories)
├── mutable_train.py      # THE optimization target (auto-modified)
├── policy.py             # AI policy: proposes code changes
├── runner.py             # Applies patches, runs training, calls judge
├── judge.py              # Compares results: keep / discard / crash
├── generate.py           # Generate text with the current best model
├── constitution.md       # Immutable system rules
├── requirements.txt      # Python dependencies
├── baseline/             # Current best version of mutable_train.py
├── data/                 # Cached training data (auto-generated)
├── memory.jsonl          # Experiment history log
└── dashboard/            # Local web dashboard (FastAPI + static)
```

## The Model

The optimization target is a **tiny decoder-only Transformer** — the same architecture family as GPT, but scaled down to run on personal computers:

| Parameter | Default Value |
|-----------|--------------|
| Layers | 4 |
| Attention heads | 4 |
| Embedding dimension | 128 |
| Context length | 256 tokens |
| Training budget | 500 steps per experiment |
| Total parameters | ~1M (varies as architecture evolves) |

**Training data**: [TinyStories](https://huggingface.co/datasets/roneneldan/TinyStories) — a dataset of short children's stories generated by GPT-3.5/4, designed for training and evaluating small language models.

**What the model can do**: After training, the model can generate simple short stories in English. It won't write essays or answer questions — it's a minimal language model meant to be a measurable optimization target, not a production tool.

**What evolves**: The system automatically modifies model architecture (depth, width, heads), optimizer settings (learning rate, weight decay), training schedule (warmup, decay), and batch configuration. The evaluation metric (`val_loss` on a fixed validation set) stays frozen so improvements are real and comparable.

## How the Automatic Evolution Works

The system runs an endless loop. Each round is one complete experiment — fully automatic, no human input needed:

### Step 1: Analyze history and propose a change (`policy.py`)

The AI reads the last 15–20 experiment records from `memory.jsonl` and the current training script code. It analyzes what types of changes led to improvements, what failed, whether val_loss is still improving or has plateaued, and what hasn't been tried yet.

Based on this analysis, it generates a **patch plan** — a specific find-and-replace modification to `mutable_train.py`. For example, it might change `n_layer = 4` to `n_layer = 6`, rewrite the learning rate schedule, or introduce a new normalization technique.

If no AI API key is configured, the system falls back to built-in heuristic rules that randomly explore hyperparameter changes.

### Step 2: Apply the change and train (`runner.py`)

The runner takes the current best version of `mutable_train.py` (stored in `baseline/`), applies the proposed modification, and launches training as a subprocess. Training runs for exactly **500 steps** — this fixed budget ensures every experiment is fairly comparable regardless of what the AI changed.

### Step 3: Judge the result (`judge.py`)

After training completes, the judge compares the new `val_loss` against the baseline:
- **Keep** — val_loss improved, or val_loss is similar but the model has fewer parameters
- **Discard** — val_loss got worse or didn't meaningfully improve
- **Crash** — training failed (syntax error, NaN, out of memory, timeout)

### Step 4: Update or rollback

If **keep**: the modified script becomes the new baseline. Future experiments build on this improved version. If **discard** or **crash**: the modification is thrown away and the script is restored. Nothing is lost.

### Step 5: Record and learn

Every experiment is logged to `memory.jsonl` with full details: what was changed, the hypothesis, val_loss, training time, memory usage, verdict, and a short lesson (e.g., "architecture change was beneficial" or "optimizer change caused crash — avoid this direction").

The next round, the AI reads this updated history and makes a more informed decision. Over many rounds, the system builds up a memory of what works and what doesn't, gradually converging toward better model configurations and architectures.

## Background

This project is inspired by Andrej Karpathy's [autoresearch](https://github.com/karpathy/autoresearch) — the idea that AI can automate the research cycle: propose a change, run an experiment, evaluate results, keep or discard, and repeat. SelfEvo applies this concept at a smaller scale so it can run on any personal computer without an NVIDIA GPU.

## Documentation

- [Constitution](constitution.md) — Immutable system rules
- [PRD](PRD/self_evo_tiny_llm_prd_v_1.md) — Product requirements document

Chinese versions: [AGENTS_zh.md](AGENTS_zh.md) | [constitution_zh.md](constitution_zh.md) | [PRD (中文)](PRD/self_evo_tiny_llm_prd_v_1_zh.md)

## License

[MIT](LICENSE)
