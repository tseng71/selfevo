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
    save_state(state)
    return {"ok": True, "settings": settings}


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
        block_size = model.pos_emb.weight.shape[0]

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
    """Retrain the model from the current baseline and save a new checkpoint."""
    import subprocess
    ckpt_path = PROJECT_DIR / "checkpoint.pt"

    # Delete old checkpoint to force retrain
    if ckpt_path.exists():
        ckpt_path.unlink()

    # Clear cache
    _model_cache["model"] = None

    # Run generate.py to create checkpoint (it auto-trains if no checkpoint)
    try:
        result = subprocess.run(
            [sys.executable, str(PROJECT_DIR / "generate.py"),
             "--prompt", "test", "--num", "1", "--tokens", "1"],
            capture_output=True, text=True, timeout=300,
            cwd=str(PROJECT_DIR),
        )
        if ckpt_path.exists():
            return {"ok": True, "message": "Model retrained and checkpoint saved"}
        return {"ok": False, "error": result.stderr[-500:] if result.stderr else "Unknown error"}
    except subprocess.TimeoutExpired:
        return {"ok": False, "error": "Training timed out"}
    except Exception as e:
        return {"ok": False, "error": str(e)}


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
