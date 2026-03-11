# SelfEvo - AGENTS.md

## Project Overview

SelfEvo is a local self-evolving experiment system that runs on personal computers (Apple Silicon Mac, Windows, Linux). Its sole optimization target is `mutable_train.py`.

## Project Structure

```
selfevo/
├── prepare.py            # Data preparation (TinyStories)
├── mutable_train.py      # Sole optimization target: tiny Transformer training script
├── policy.py             # Policy module: generates patch plans
├── runner.py             # Execution module: applies patches, runs training, calls judge
├── judge.py              # Judging module: keep / discard / crash
├── main.py               # Entry point: launches dashboard + experiment loop
├── generate.py           # Text generation: load best model and generate samples
├── constitution.md       # Supreme rules (immutable)
├── requirements.txt      # Python dependencies
├── data/                 # Data cache (tokenizer, train.bin, val.bin)
├── baseline/             # Current best version
├── memory.jsonl          # Experiment history
├── state.json            # Runtime state (dashboard IPC)
└── dashboard/
    ├── app.py            # FastAPI backend
    └── static/           # Frontend files (HTML/JS/CSS)
```

## Development Constraints

1. **Only modify mutable_train.py**: The system's automatic evolution only acts on this single file
2. **Do not modify evaluation logic**: The val_loss computation and validation data must remain frozen
3. **Cross-platform compatible**: Supports Apple Silicon (MPS), NVIDIA GPU (CUDA), and CPU-only mode
4. **Local execution**: No public deployment, no account system required
5. **Complete logging**: Every experiment round must be recorded in memory.jsonl

## Key Workflow

1. `prepare.py` prepares data
2. `runner.py` calls `policy.py` to generate a patch plan
3. `runner.py` applies the patch to `mutable_train.py`
4. A subprocess runs the training
5. `judge.py` compares the results
6. keep → update baseline; discard/crash → rollback
7. Write to memory.jsonl
8. Dashboard displays the results

## Tech Stack

- Python 3.10+
- PyTorch (MPS / CUDA / CPU backend)
- FastAPI + uvicorn (dashboard)
- Chart.js (frontend charts)
- tokenizers (BPE tokenization)
- datasets (HuggingFace data loading)
- google-generativeai (Gemini API for AI-powered policy)
