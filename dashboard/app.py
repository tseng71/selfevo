"""
SelfEvo - Dashboard Backend (FastAPI)
Provides REST API for the visualization panel and serves static files.
"""

import json
import os
import sys
from collections import Counter
from datetime import datetime
from pathlib import Path

from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

# Add parent to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

PROJECT_DIR = Path(__file__).parent.parent
MEMORY_PATH = PROJECT_DIR / "memory.jsonl"
STATE_PATH = PROJECT_DIR / "state.json"
BASELINE_SCRIPT = PROJECT_DIR / "baseline" / "mutable_train.py"
MUTABLE_SCRIPT = PROJECT_DIR / "mutable_train.py"
DATA_DIR = PROJECT_DIR / "data"
STATIC_DIR = Path(__file__).parent / "static"

app = FastAPI(title="SelfEvo Dashboard")


@app.on_event("startup")
def load_saved_api_keys():
    """Load API keys from state.json into environment on startup."""
    state = load_state()
    saved_keys = state.get("api_keys", {})
    key_map = {
        "gemini": "GEMINI_API_KEY",
        "openai": "OPENAI_API_KEY",
        "claude": "ANTHROPIC_API_KEY",
    }
    for provider, env_name in key_map.items():
        if provider in saved_keys and saved_keys[provider]:
            if not os.environ.get(env_name):
                os.environ[env_name] = saved_keys[provider]
                print(f"[dashboard] Loaded {provider} API key from state.json")


def load_memory():
    """Load all experiment records."""
    if not MEMORY_PATH.exists():
        return []
    records = []
    with open(MEMORY_PATH) as f:
        for line in f:
            line = line.strip()
            if line:
                try:
                    records.append(json.loads(line))
                except json.JSONDecodeError:
                    pass
    return records


def load_state():
    """Load current system state."""
    if not STATE_PATH.exists():
        return {"status": "idle", "total_experiments": 0}
    try:
        return json.loads(STATE_PATH.read_text())
    except (json.JSONDecodeError, FileNotFoundError):
        return {"status": "idle", "total_experiments": 0}


def save_state(state):
    """Save system state."""
    STATE_PATH.write_text(json.dumps(state, indent=2))


# === API Endpoints ===

@app.get("/api/status")
def get_status():
    """Get current system status."""
    state = load_state()
    records = load_memory()

    # Compute stats
    keeps = sum(1 for r in records if r.get("status") == "keep")
    discards = sum(1 for r in records if r.get("status") == "discard")
    crashes = sum(1 for r in records if r.get("status") == "crash")

    # Best val_loss
    best_val_loss = None
    best_experiment = None
    for r in records:
        if r.get("status") == "keep" and r.get("val_loss") is not None:
            if best_val_loss is None or r["val_loss"] < best_val_loss:
                best_val_loss = r["val_loss"]
                best_experiment = r.get("experiment_id")

    # Today's experiments
    today = datetime.now().strftime("%Y-%m-%d")
    today_count = sum(1 for r in records if r.get("timestamp", "").startswith(today))

    # Current phase
    phase = "idle"
    if records:
        recent = records[-5:]
        recent_crashes = sum(1 for r in recent if r.get("status") == "crash")
        recent_keeps = sum(1 for r in recent if r.get("status") == "keep")
        if recent_crashes >= 3:
            phase = "repair"
        elif recent_keeps >= 2:
            phase = "exploitation"
        elif len(records) <= 3:
            phase = "baseline"
        else:
            phase = "exploration"

    # Data ready
    data_ready = (DATA_DIR / "train.bin").exists() and (DATA_DIR / "val.bin").exists()

    return {
        "status": state.get("status", "idle"),
        "data_ready": data_ready,
        "total_experiments": len(records),
        "today_experiments": today_count,
        "keeps": keeps,
        "discards": discards,
        "crashes": crashes,
        "best_val_loss": best_val_loss,
        "best_experiment": best_experiment,
        "phase": phase,
        "last_experiment": records[-1] if records else None,
        "last_update": state.get("last_update"),
    }


@app.get("/api/experiments")
def get_experiments(
    status: str = None,
    experiment_class: str = None,
    limit: int = 100,
    offset: int = 0,
):
    """Get experiment history with optional filtering."""
    records = load_memory()

    # Filter
    if status:
        records = [r for r in records if r.get("status") == status]
    if experiment_class:
        records = [r for r in records if r.get("experiment_class") == experiment_class]

    # Reverse chronological
    records.reverse()

    total = len(records)
    records = records[offset : offset + limit]

    return {"total": total, "offset": offset, "limit": limit, "experiments": records}


@app.get("/api/trends")
def get_trends():
    """Get aggregated trend data for charts."""
    records = load_memory()

    # val_loss over time
    val_loss_trend = []
    keep_rate_trend = []
    crash_rate_trend = []

    window = 5
    for i, r in enumerate(records):
        val_loss_trend.append({
            "index": i,
            "experiment_id": r.get("experiment_id"),
            "val_loss": r.get("val_loss"),
            "status": r.get("status"),
            "timestamp": r.get("timestamp"),
        })

        # Rolling rates
        if i >= window - 1:
            recent = records[i - window + 1 : i + 1]
            keeps = sum(1 for x in recent if x.get("status") == "keep")
            crashes = sum(1 for x in recent if x.get("status") == "crash")
            keep_rate_trend.append({"index": i, "rate": keeps / window})
            crash_rate_trend.append({"index": i, "rate": crashes / window})

    # Success rate per experiment class
    class_stats = {}
    for r in records:
        cls = r.get("experiment_class", "unknown")
        if cls not in class_stats:
            class_stats[cls] = {"total": 0, "keeps": 0, "crashes": 0}
        class_stats[cls]["total"] += 1
        if r.get("status") == "keep":
            class_stats[cls]["keeps"] += 1
        elif r.get("status") == "crash":
            class_stats[cls]["crashes"] += 1

    class_success = [
        {
            "class": cls,
            "total": s["total"],
            "keep_rate": s["keeps"] / s["total"] if s["total"] > 0 else 0,
            "crash_rate": s["crashes"] / s["total"] if s["total"] > 0 else 0,
        }
        for cls, s in class_stats.items()
    ]

    # Best val_loss progression (only keeps)
    best_progression = []
    current_best = None
    for r in records:
        if r.get("status") == "keep" and r.get("val_loss") is not None:
            if current_best is None or r["val_loss"] < current_best:
                current_best = r["val_loss"]
            best_progression.append({
                "experiment_id": r.get("experiment_id"),
                "val_loss": current_best,
                "timestamp": r.get("timestamp"),
            })

    return {
        "val_loss_trend": val_loss_trend,
        "keep_rate_trend": keep_rate_trend,
        "crash_rate_trend": crash_rate_trend,
        "class_success": class_success,
        "best_progression": best_progression,
    }


@app.get("/api/baseline")
def get_baseline():
    """Get current baseline information."""
    baseline_result = None
    records = load_memory()
    for r in reversed(records):
        if r.get("status") == "keep":
            baseline_result = r
            break

    baseline_code = None
    if BASELINE_SCRIPT.exists():
        baseline_code = BASELINE_SCRIPT.read_text()

    return {
        "exists": BASELINE_SCRIPT.exists(),
        "result": baseline_result,
        "code_preview": baseline_code[:2000] if baseline_code else None,
    }


@app.get("/api/compare/{experiment_id}")
def compare_experiment(experiment_id: str):
    """Compare a specific experiment with the baseline."""
    records = load_memory()

    # Find the experiment
    experiment = None
    for r in records:
        if r.get("experiment_id") == experiment_id:
            experiment = r
            break

    if not experiment:
        raise HTTPException(status_code=404, detail="Experiment not found")

    # Find baseline at that time (last keep before this experiment)
    baseline = None
    for r in records:
        if r.get("experiment_id") == experiment_id:
            break
        if r.get("status") == "keep":
            baseline = r

    return {
        "experiment": experiment,
        "baseline": baseline,
    }


@app.post("/api/control/{action}")
def control_action(action: str):
    """Handle control actions from the dashboard."""
    state = load_state()

    if action == "start":
        state["status"] = "running"
        state["command"] = "start"
        save_state(state)
        return {"ok": True, "message": "Started experiment loop"}

    elif action == "pause":
        state["status"] = "paused"
        state["command"] = "pause"
        save_state(state)
        return {"ok": True, "message": "Paused experiment loop"}

    elif action == "step":
        state["command"] = "step"
        save_state(state)
        return {"ok": True, "message": "Running single experiment"}

    elif action == "rollback":
        if BASELINE_SCRIPT.exists():
            import shutil
            shutil.copy2(BASELINE_SCRIPT, MUTABLE_SCRIPT)
            state["status"] = "idle"
            state["message"] = "Rolled back to best baseline"
            state["last_update"] = datetime.now().isoformat()
            save_state(state)
            return {"ok": True, "message": "Rolled back to best baseline"}
        return {"ok": False, "message": "No baseline found"}

    elif action == "lock":
        state["locked"] = True
        save_state(state)
        return {"ok": True, "message": "Baseline locked"}

    elif action == "unlock":
        state["locked"] = False
        save_state(state)
        return {"ok": True, "message": "Baseline unlocked"}

    else:
        raise HTTPException(status_code=400, detail=f"Unknown action: {action}")


@app.post("/api/settings")
def update_settings(settings: dict):
    """Update system settings."""
    state = load_state()
    if "allow_high_risk" in settings:
        state["allow_high_risk"] = settings["allow_high_risk"]
    if "allow_large_changes" in settings:
        state["allow_large_changes"] = settings["allow_large_changes"]
    if "ai_provider" in settings:
        state["ai_provider"] = settings["ai_provider"] or None
    if "ai_model" in settings:
        state["ai_model"] = settings["ai_model"] or None
    save_state(state)
    return {"ok": True, "settings": settings}


@app.get("/api/ai_config")
def get_ai_config():
    """Get current AI provider/model configuration."""
    import os as _os
    state = load_state()
    saved_keys = state.get("api_keys", {})

    key_map = {
        "gemini": "GEMINI_API_KEY",
        "openai": "OPENAI_API_KEY",
        "claude": "ANTHROPIC_API_KEY",
    }

    available = []
    key_sources = {}
    for provider, env_name in key_map.items():
        if _os.environ.get(env_name):
            available.append(provider)
            # Determine source: dashboard-saved vs environment
            if saved_keys.get(provider) and saved_keys[provider] == _os.environ.get(env_name):
                key_sources[provider] = "dashboard"
            else:
                key_sources[provider] = "env"

    return {
        "available_providers": available,
        "current_provider": state.get("ai_provider"),
        "current_model": state.get("ai_model"),
        "defaults": {
            "gemini": "gemini-3.1-pro-preview",
            "openai": "gpt-5.4",
            "claude": "claude-opus-4-6",
        },
        "key_sources": key_sources,
    }


@app.post("/api/save_key")
def save_api_key(data: dict):
    """Save an API key from the dashboard UI."""
    provider = data.get("provider", "")
    key = data.get("key", "").strip()
    if not provider or not key:
        raise HTTPException(status_code=400, detail="Missing provider or key")

    key_map = {
        "gemini": "GEMINI_API_KEY",
        "openai": "OPENAI_API_KEY",
        "claude": "ANTHROPIC_API_KEY",
    }
    env_name = key_map.get(provider)
    if not env_name:
        raise HTTPException(status_code=400, detail=f"Unknown provider: {provider}")

    # Set in environment for immediate use
    os.environ[env_name] = key

    # Persist in state.json
    state = load_state()
    if "api_keys" not in state:
        state["api_keys"] = {}
    state["api_keys"][provider] = key
    save_state(state)

    return {"ok": True, "message": f"{provider} API key saved"}


# === Text Generation (Playground) ===

# Cache the model in memory
_model_cache = {"model": None, "tokenizer": None, "device": None, "config": None}


def _load_model_for_generation():
    """Load or reload the model for text generation."""
    import importlib.util
    import torch

    ckpt_path = PROJECT_DIR / "checkpoint.pt"
    meta_path = DATA_DIR / "meta.json"
    tokenizer_path = DATA_DIR / "tokenizer.json"

    if not meta_path.exists() or not BASELINE_SCRIPT.exists():
        return None, None, None, "Model or data not ready"

    with open(meta_path) as f:
        meta = json.load(f)
    vocab_size = meta["vocab_size"]

    # Check if baseline changed since last load
    baseline_mtime = BASELINE_SCRIPT.stat().st_mtime
    if _model_cache["model"] is not None and _model_cache.get("_mtime") == baseline_mtime:
        return _model_cache["model"], _model_cache["tokenizer"], _model_cache["device"], None

    # Load model definition from baseline
    baseline_code = BASELINE_SCRIPT.read_text()
    mod_dict = {"__file__": str(BASELINE_SCRIPT), "__name__": "baseline_train"}
    exec(compile(baseline_code, str(BASELINE_SCRIPT), "exec"), mod_dict)

    model = mod_dict["TinyTransformer"](vocab_size)

    # Device
    if torch.backends.mps.is_available():
        device = torch.device("mps")
    elif torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")

    # Load checkpoint or train
    if ckpt_path.exists():
        state_dict = torch.load(ckpt_path, map_location="cpu", weights_only=True)
        try:
            model.load_state_dict(state_dict)
        except RuntimeError:
            # Architecture changed, need to retrain
            return None, None, None, "Checkpoint incompatible with current model. Click 'Retrain' to create a new checkpoint."

    model = model.to(device)
    model.eval()

    # Load tokenizer
    from tokenizers import Tokenizer
    tokenizer = Tokenizer.from_file(str(tokenizer_path))

    # Cache
    _model_cache["model"] = model
    _model_cache["tokenizer"] = tokenizer
    _model_cache["device"] = device
    _model_cache["_mtime"] = baseline_mtime

    return model, tokenizer, device, None


@app.post("/api/generate")
def generate_text(request: dict):
    """Generate text using the current model."""
    import torch
    import torch.nn.functional as F

    prompt = request.get("prompt", "Once upon a time")
    max_tokens = min(request.get("max_tokens", 200), 500)
    temperature = max(0.0, min(request.get("temperature", 0.8), 2.0))
    top_k = max(0, min(request.get("top_k", 40), 100))

    model, tokenizer, device, error = _load_model_for_generation()
    if error:
        return {"ok": False, "error": error}

    try:
        encoded = tokenizer.encode(prompt)
        ids = encoded.ids
        # Get block_size from baseline config (model may not have pos_emb anymore)
        import re as _re
        _bs_match = _re.search(r"^block_size\s*=\s*(\d+)", BASELINE_SCRIPT.read_text(), _re.MULTILINE)
        block_size = int(_bs_match.group(1)) if _bs_match else 256

        x = torch.tensor([ids], dtype=torch.long, device=device)
        generated_ids = list(ids)
        eot_id = tokenizer.token_to_id("<|endoftext|>")

        with torch.no_grad():
            for _ in range(max_tokens):
                x_cond = x if x.shape[1] <= block_size else x[:, -block_size:]
                logits = model(x_cond)[:, -1, :]

                if temperature > 0:
                    logits = logits / temperature
                    if top_k > 0:
                        v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                        logits[logits < v[:, [-1]]] = -float("inf")
                    probs = F.softmax(logits, dim=-1)
                    next_id = torch.multinomial(probs, num_samples=1)
                else:
                    next_id = logits.argmax(dim=-1, keepdim=True)

                generated_ids.append(next_id.item())
                x = torch.cat([x, next_id], dim=1)

                if next_id.item() == eot_id:
                    break

        text = tokenizer.decode(generated_ids)
        return {
            "ok": True,
            "text": text,
            "prompt": prompt,
            "tokens_generated": len(generated_ids) - len(ids),
            "model_params": model.count_params(),
        }

    except Exception as e:
        return {"ok": False, "error": str(e)}


@app.post("/api/retrain")
def retrain_model():
    """Retrain the model using baseline script and save checkpoint."""
    import subprocess
    import torch
    ckpt_path = PROJECT_DIR / "checkpoint.pt"

    # Delete old checkpoint and clear cache
    if ckpt_path.exists():
        ckpt_path.unlink()
    _model_cache["model"] = None

    # Run baseline/mutable_train.py which outputs JSON result,
    # then save model state_dict by loading the trained model
    try:
        # Step 1: Train by running the baseline script directly
        result = subprocess.run(
            [sys.executable, str(BASELINE_SCRIPT)],
            capture_output=True, text=True, timeout=1800,
            cwd=str(PROJECT_DIR),
        )

        # Step 2: Rebuild model with current architecture and train to save checkpoint
        baseline_code = BASELINE_SCRIPT.read_text()
        mod_dict = {"__file__": str(BASELINE_SCRIPT), "__name__": "retrain_module"}
        exec(compile(baseline_code, str(BASELINE_SCRIPT), "exec"), mod_dict)

        with open(DATA_DIR / "meta.json") as f:
            meta = json.load(f)

        # Build and train
        model = mod_dict["TinyTransformer"](meta["vocab_size"])
        device = torch.device("mps") if torch.backends.mps.is_available() else torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

        import numpy as np
        train_data = np.memmap(str(DATA_DIR / "train.bin"), dtype=np.uint16, mode="r")
        model = model.to(device)
        model.train()

        max_steps = mod_dict.get("max_steps", 500)
        batch_size = mod_dict.get("batch_size", 32)
        block_size_val = mod_dict.get("block_size", 256)
        grad_accum = mod_dict.get("gradient_accumulation_steps", 1)
        grad_clip_val = mod_dict.get("grad_clip", 1.0)
        get_lr = mod_dict["get_lr"]

        # Build optimizer same as baseline
        decay_params = [p for p in model.parameters() if p.dim() >= 2]
        nodecay_params = [p for p in model.parameters() if p.dim() < 2]
        optimizer = torch.optim.AdamW([
            {"params": decay_params, "weight_decay": float(mod_dict.get("weight_decay", 0.1))},
            {"params": nodecay_params, "weight_decay": 0.0}
        ], lr=float(mod_dict.get("learning_rate", 1e-3)),
           betas=(float(mod_dict.get("beta1", 0.9)), float(mod_dict.get("beta2", 0.95))))

        import math
        for step in range(max_steps):
            lr_val = get_lr(step)
            for pg in optimizer.param_groups:
                pg["lr"] = lr_val
            optimizer.zero_grad()
            for _ in range(grad_accum):
                ix = torch.randint(len(train_data) - block_size_val - 1, (batch_size,))
                x = torch.stack([torch.from_numpy(train_data[i:i+block_size_val].astype(np.int64)) for i in ix]).to(device)
                y = torch.stack([torch.from_numpy(train_data[i+1:i+1+block_size_val].astype(np.int64)) for i in ix]).to(device)
                logits = model(x)
                loss = torch.nn.functional.cross_entropy(logits.view(-1, logits.size(-1)), y.view(-1)) / grad_accum
                loss.backward()
            if grad_clip_val > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip_val)
            optimizer.step()

        torch.save(model.state_dict(), str(ckpt_path))
        return {"ok": True, "message": f"Model retrained ({max_steps} steps) and checkpoint saved"}

    except Exception as e:
        return {"ok": False, "error": str(e)[:500]}


@app.get("/api/model_info")
def get_model_info():
    """Get info about the current model."""
    ckpt_path = PROJECT_DIR / "checkpoint.pt"
    has_checkpoint = ckpt_path.exists()

    config = {}
    if BASELINE_SCRIPT.exists():
        import re
        code = BASELINE_SCRIPT.read_text()
        for key in ["n_layer", "n_head", "n_embd", "block_size", "dropout"]:
            m = re.search(rf"^{key}\s*=\s*(.+)", code, re.MULTILINE)
            if m:
                config[key] = m.group(1).strip()

    return {
        "has_checkpoint": has_checkpoint,
        "checkpoint_size_mb": round(ckpt_path.stat().st_size / 1024 / 1024, 1) if has_checkpoint else None,
        "config": config,
    }


# Serve static files
app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")


@app.get("/")
def serve_index():
    """Serve the main dashboard page."""
    return FileResponse(str(STATIC_DIR / "index.html"))
