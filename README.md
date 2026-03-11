# SelfEvo - Self-Evolving AI Training System

SelfEvo is a local self-evolving experiment system inspired by Andrej Karpathy's [autoresearch](https://x.com/kaboroevich/status/1927439207498973645) concept. It automatically and iteratively improves a tiny language model training script through an autonomous loop of code modification, training, evaluation, and selection.

## What It Does

SelfEvo wraps a tiny decoder-only Transformer (trained on [TinyStories](https://huggingface.co/datasets/roneneldan/TinyStories)) in a closed-loop evolution cycle:

1. **Propose** — An AI policy (powered by Gemini) analyzes experiment history and proposes code modifications to the training script
2. **Train** — The modified script runs a fixed-budget training session
3. **Judge** — Results are compared against the current best (baseline) using `val_loss`
4. **Keep or Discard** — Better results update the baseline; worse results are rolled back
5. **Learn** — All experiments are logged, informing future proposals

A local web dashboard lets you observe progress, review experiments, and intervene — no programming knowledge required.

## Key Features

- **Single optimization target**: Only `mutable_train.py` is modified — evaluation logic is frozen
- **AI-powered policy**: Uses Gemini API to generate intelligent code modifications (falls back to heuristic rules if API is unavailable)
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

### 2. Set up your Gemini API key

The AI-powered policy module requires a [Google Gemini API key](https://aistudio.google.com/apikey). Without it, the system falls back to heuristic-based mutations.

```bash
export GEMINI_API_KEY="your-api-key-here"
```

On Windows:
```cmd
set GEMINI_API_KEY=your-api-key-here
```

### 3. Prepare data

```bash
python prepare.py
```

This downloads TinyStories, trains a BPE tokenizer, and creates cached binary files.

### 4. Run

```bash
python main.py
```

Then open the dashboard URL shown in the terminal (default: http://localhost:8000) to watch the experiments.

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

## How It Works

SelfEvo takes the idea of "AI doing research" and narrows it down to a minimal, verifiable loop:

- **One file to optimize**: `mutable_train.py` — a tiny Transformer training script
- **One metric to chase**: `val_loss` on a fixed validation set
- **One fixed budget**: 500 training steps per experiment
- **One clear rule**: if it's better, keep it; if not, discard it

The AI policy reads experiment history, identifies what worked and what didn't, and proposes the next modification. Over many rounds, the system accumulates experience and (ideally) converges toward better configurations and architectures.

## Background

This project is inspired by the "autoresearch" direction discussed by Andrej Karpathy and others — the idea that AI can automate the "propose hypothesis → run experiment → analyze results → iterate" cycle of research. SelfEvo is a concrete, minimal implementation of this idea: small enough to run on a personal computer, constrained enough to be verifiable, and transparent enough for non-programmers to observe and control.

## Documentation

- [Constitution](constitution.md) — Immutable system rules
- [PRD](PRD/self_evo_tiny_llm_prd_v_1.md) — Product requirements document

Chinese versions: [AGENTS_zh.md](AGENTS_zh.md) | [constitution_zh.md](constitution_zh.md) | [PRD (中文)](PRD/self_evo_tiny_llm_prd_v_1_zh.md)

## License

[MIT](LICENSE)
