"""
SelfEvo - Runner Module
Orchestrates a single experiment round: patch, train, judge, update.
"""

import json
import os
import re
import shutil
import subprocess
import sys
import time
import uuid
from datetime import datetime
from pathlib import Path

from judge import judge
from policy import generate_patch_plan, load_history

PROJECT_DIR = Path(__file__).parent
MUTABLE_SCRIPT = PROJECT_DIR / "mutable_train.py"
BASELINE_DIR = PROJECT_DIR / "baseline"
BASELINE_SCRIPT = BASELINE_DIR / "mutable_train.py"
MEMORY_PATH = PROJECT_DIR / "memory.jsonl"
STATE_PATH = PROJECT_DIR / "state.json"

TRAIN_TIMEOUT = 1800  # seconds (30 min, generous budget for code-level experiments)


def init_baseline():
    """Initialize baseline from mutable_train.py if not exists."""
    BASELINE_DIR.mkdir(parents=True, exist_ok=True)
    if not BASELINE_SCRIPT.exists():
        shutil.copy2(MUTABLE_SCRIPT, BASELINE_SCRIPT)
        print("[runner] Initialized baseline from mutable_train.py")


def load_baseline_result():
    """Load the baseline result from the most recent 'keep' in memory."""
    if not MEMORY_PATH.exists():
        return None
    records = []
    with open(MEMORY_PATH) as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    # Find most recent keep
    for r in reversed(records):
        if r.get("status") == "keep":
            return {
                "val_loss": r.get("val_loss"),
                "num_params": r.get("num_params", 0),
                "status": "ok",
            }
    return None


def apply_mutations(source_code, mutations):
    """Apply find/replace mutations to the source code."""
    modified = source_code
    applied = []
    for m in mutations:
        find = m["find"]
        replace = m["replace"]
        if find and find in modified:
            modified = modified.replace(find, replace, 1)
            applied.append(f"{find} -> {replace}")
        elif find:
            # Try regex-based matching for partial patterns like "n_layer = "
            pattern = re.escape(find) + r"[^\n]*"
            if re.search(pattern, modified):
                modified = re.sub(pattern, replace, modified, count=1)
                applied.append(f"{find}* -> {replace}")
    return modified, applied


def run_training(script_path, timeout=TRAIN_TIMEOUT):
    """Run the training script as a subprocess and capture output."""
    try:
        result = subprocess.run(
            [sys.executable, str(script_path)],
            capture_output=True,
            text=True,
            timeout=timeout,
            cwd=str(PROJECT_DIR),
        )

        stdout = result.stdout.strip()
        stderr = result.stderr.strip()

        # Check for syntax errors
        if result.returncode != 0:
            if "SyntaxError" in stderr:
                return {"status": "syntax", "error": stderr[-500:]}
            if "MemoryError" in stderr or "out of memory" in stderr.lower():
                return {"status": "oom", "error": stderr[-500:]}
            return {"status": "error", "error": stderr[-500:]}

        # Parse JSON from last line of stdout
        lines = stdout.split("\n")
        for line in reversed(lines):
            line = line.strip()
            if line.startswith("{"):
                try:
                    return json.loads(line)
                except json.JSONDecodeError:
                    pass

        return {"status": "error", "error": "No valid JSON output found"}

    except subprocess.TimeoutExpired:
        return {"status": "timeout", "error": f"Training exceeded {timeout}s timeout"}
    except Exception as e:
        return {"status": "error", "error": str(e)[:500]}


def write_memory(record):
    """Append a record to memory.jsonl."""
    with open(MEMORY_PATH, "a") as f:
        f.write(json.dumps(record) + "\n")


def update_state(state_updates):
    """Update the state.json file for dashboard IPC."""
    state = {}
    if STATE_PATH.exists():
        try:
            state = json.loads(STATE_PATH.read_text())
        except (json.JSONDecodeError, FileNotFoundError):
            state = {}
    state.update(state_updates)
    STATE_PATH.write_text(json.dumps(state, indent=2))


def get_experiment_id():
    """Generate a unique experiment ID."""
    return f"exp-{datetime.now().strftime('%Y%m%d-%H%M%S')}-{uuid.uuid4().hex[:6]}"


def count_experiments():
    """Count total experiments."""
    if not MEMORY_PATH.exists():
        return 0
    count = 0
    with open(MEMORY_PATH) as f:
        for line in f:
            if line.strip():
                count += 1
    return count


def run_single_experiment(require_approval=False):
    """
    Execute one full experiment round.

    Returns:
        dict: The experiment record.
    """
    experiment_id = get_experiment_id()
    timestamp = datetime.now().isoformat()

    # Ensure baseline exists
    init_baseline()

    # Update state: running
    update_state({
        "status": "running",
        "current_experiment": experiment_id,
        "last_update": timestamp,
    })

    # Step 1: Generate patch plan
    history = load_history()
    patch_plan = generate_patch_plan(history)

    # Check if high risk and approval needed
    if require_approval and patch_plan.get("high_risk", False):
        update_state({
            "status": "awaiting_approval",
            "pending_plan": patch_plan,
            "current_experiment": experiment_id,
        })
        return {"status": "awaiting_approval", "plan": patch_plan, "experiment_id": experiment_id}

    # Step 2: Apply patch
    baseline_code = BASELINE_SCRIPT.read_text()

    if patch_plan.get("is_revert"):
        # Revert: just use baseline as-is
        modified_code = baseline_code
        applied = ["revert to baseline"]
        patch_summary = "Revert to baseline (repair)"
    else:
        modified_code, applied = apply_mutations(baseline_code, patch_plan.get("mutations", []))
        patch_summary = "; ".join(applied) if applied else "no changes applied"

    # Write modified script
    MUTABLE_SCRIPT.write_text(modified_code)

    # Step 3: Run training
    update_state({"status": "training", "current_experiment": experiment_id})
    train_result = run_training(MUTABLE_SCRIPT)

    # Step 4: Judge
    baseline_result = load_baseline_result()
    verdict = judge(train_result, baseline_result)

    # Step 5: Update baseline or rollback
    latest_ckpt = PROJECT_DIR / "_latest_checkpoint.pt"
    if verdict["verdict"] == "keep":
        shutil.copy2(MUTABLE_SCRIPT, BASELINE_SCRIPT)
        # Copy checkpoint for playground
        if latest_ckpt.exists():
            shutil.copy2(latest_ckpt, PROJECT_DIR / "checkpoint.pt")
    else:
        # Rollback: restore baseline
        shutil.copy2(BASELINE_SCRIPT, MUTABLE_SCRIPT)
    # Clean up temp checkpoint
    if latest_ckpt.exists():
        latest_ckpt.unlink()

    # Step 6: Write memory
    record = {
        "experiment_id": experiment_id,
        "timestamp": timestamp,
        "parent_version": f"exp-{count_experiments():04d}",
        "patch_summary": patch_summary,
        "experiment_class": patch_plan.get("experiment_class", "unknown"),
        "target_zone": patch_plan.get("target_zone", "CONFIG"),
        "hypothesis": patch_plan.get("hypothesis", ""),
        "source": patch_plan.get("source", "unknown"),
        "ai_analysis": patch_plan.get("analysis", {}),
        "val_loss": train_result.get("val_loss"),
        "train_time_sec": train_result.get("train_time_sec"),
        "total_time_sec": train_result.get("total_time_sec"),
        "peak_mem_mb": train_result.get("peak_mem_mb"),
        "num_params": train_result.get("num_params"),
        "num_steps": train_result.get("num_steps"),
        "status": verdict["verdict"],
        "failure_type": train_result.get("status") if verdict["verdict"] == "crash" else None,
        "judge_reason": verdict["reason"],
        "lesson": _extract_lesson(verdict, patch_plan),
    }
    write_memory(record)

    # Step 7: Update state
    # Find current best
    best_val_loss = None
    best_experiment = None
    if MEMORY_PATH.exists():
        with open(MEMORY_PATH) as f:
            for line in f:
                line = line.strip()
                if line:
                    r = json.loads(line)
                    if r.get("status") == "keep" and r.get("val_loss") is not None:
                        if best_val_loss is None or r["val_loss"] < best_val_loss:
                            best_val_loss = r["val_loss"]
                            best_experiment = r["experiment_id"]

    # Preserve current status (don't override "running" set by main.py)
    current_state = {}
    if STATE_PATH.exists():
        try:
            current_state = json.loads(STATE_PATH.read_text())
        except (json.JSONDecodeError, FileNotFoundError):
            pass
    current_status = current_state.get("status", "idle")

    update_state({
        "status": current_status,
        "last_experiment": record,
        "best_val_loss": best_val_loss,
        "best_experiment": best_experiment,
        "total_experiments": count_experiments(),
        "last_update": datetime.now().isoformat(),
    })

    return record


def _extract_lesson(verdict, patch_plan):
    """Extract a short lesson from the experiment outcome."""
    cls = patch_plan.get("experiment_class", "unknown")
    v = verdict["verdict"]
    if v == "keep":
        return f"{cls} change was beneficial: {patch_plan.get('hypothesis', '')}"
    elif v == "crash":
        return f"{cls} change caused crash - avoid this direction"
    else:
        return f"{cls} change did not improve results"


def rollback_to_best():
    """Rollback mutable_train.py to the current baseline."""
    if BASELINE_SCRIPT.exists():
        shutil.copy2(BASELINE_SCRIPT, MUTABLE_SCRIPT)
        update_state({
            "status": "idle",
            "last_update": datetime.now().isoformat(),
            "message": "Rolled back to best baseline",
        })
        return True
    return False


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="SelfEvo Runner")
    parser.add_argument("--single", action="store_true", help="Run a single experiment")
    parser.add_argument("--continuous", type=int, default=0, help="Run N experiments continuously")
    parser.add_argument("--rollback", action="store_true", help="Rollback to best baseline")
    args = parser.parse_args()

    if args.rollback:
        rollback_to_best()
        print("[runner] Rolled back to best baseline.")
    elif args.single:
        record = run_single_experiment()
        print(json.dumps(record, indent=2))
    elif args.continuous > 0:
        for i in range(args.continuous):
            print(f"\n[runner] === Experiment {i + 1}/{args.continuous} ===")
            record = run_single_experiment()
            print(f"  Status: {record.get('status')}")
            print(f"  val_loss: {record.get('val_loss')}")
            print(f"  Reason: {record.get('judge_reason')}")
    else:
        parser.print_help()
