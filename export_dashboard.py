"""
Export memory.jsonl to docs/data.json for GitHub Pages static dashboard.
Also auto-pushes to GitHub if --push flag is set.
"""

import json
import subprocess
import sys
from collections import Counter
from datetime import datetime
from pathlib import Path

PROJECT_DIR = Path(__file__).parent
MEMORY_PATH = PROJECT_DIR / "memory.jsonl"
DOCS_DIR = PROJECT_DIR / "docs"
OUTPUT_PATH = DOCS_DIR / "data.json"


def load_memory():
    if not MEMORY_PATH.exists():
        return []
    records = []
    with open(MEMORY_PATH) as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records


def compute_status(records):
    keeps = sum(1 for r in records if r.get("status") == "keep")
    discards = sum(1 for r in records if r.get("status") == "discard")
    crashes = sum(1 for r in records if r.get("status") == "crash")

    best_vl = None
    best_exp = None
    for r in records:
        vl = r.get("val_loss")
        if r.get("status") == "keep" and vl is not None:
            if best_vl is None or vl < best_vl:
                best_vl = vl
                best_exp = r.get("experiment_id")

    today = datetime.now().strftime("%Y-%m-%d")
    today_count = sum(1 for r in records if r.get("timestamp", "").startswith(today))

    last = records[-1] if records else None

    return {
        "total_experiments": len(records),
        "keeps": keeps,
        "discards": discards,
        "crashes": crashes,
        "best_val_loss": best_vl,
        "best_experiment": best_exp,
        "today_experiments": today_count,
        "last_experiment": last,
        "last_updated": datetime.now().isoformat(),
    }


def compute_trends(records):
    # val_loss trend
    val_loss_trend = []
    for r in records:
        val_loss_trend.append({
            "experiment_id": r.get("experiment_id"),
            "val_loss": r.get("val_loss"),
            "status": r.get("status"),
        })

    # Best progression
    best_progression = []
    best_vl = float("inf")
    for r in records:
        vl = r.get("val_loss")
        if r.get("status") == "keep" and vl is not None and vl < best_vl:
            best_vl = vl
            best_progression.append({
                "experiment_id": r.get("experiment_id"),
                "val_loss": vl,
            })

    # Rolling rates (window=5)
    keep_rate_trend = []
    crash_rate_trend = []
    window = 5
    for i in range(len(records)):
        start = max(0, i - window + 1)
        chunk = records[start:i + 1]
        keeps = sum(1 for r in chunk if r.get("status") == "keep")
        crashes = sum(1 for r in chunk if r.get("status") == "crash")
        n = len(chunk)
        keep_rate_trend.append({"index": i, "rate": keeps / n if n else 0})
        crash_rate_trend.append({"index": i, "rate": crashes / n if n else 0})

    # Class success rates
    class_total = Counter()
    class_keep = Counter()
    class_crash = Counter()
    for r in records:
        cls = r.get("experiment_class", "unknown")
        class_total[cls] += 1
        if r.get("status") == "keep":
            class_keep[cls] += 1
        elif r.get("status") == "crash":
            class_crash[cls] += 1

    class_success = []
    for cls in sorted(class_total.keys()):
        total = class_total[cls]
        class_success.append({
            "class": cls,
            "total": total,
            "keep_rate": class_keep[cls] / total if total else 0,
            "crash_rate": class_crash[cls] / total if total else 0,
        })

    return {
        "val_loss_trend": val_loss_trend,
        "best_progression": best_progression,
        "keep_rate_trend": keep_rate_trend,
        "crash_rate_trend": crash_rate_trend,
        "class_success": class_success,
    }


def export():
    records = load_memory()
    data = {
        "status": compute_status(records),
        "trends": compute_trends(records),
        "experiments": [{
            "experiment_id": r.get("experiment_id"),
            "timestamp": r.get("timestamp"),
            "experiment_class": r.get("experiment_class"),
            "hypothesis": r.get("hypothesis"),
            "patch_summary": r.get("patch_summary"),
            "val_loss": r.get("val_loss"),
            "train_time_sec": r.get("train_time_sec"),
            "peak_mem_mb": r.get("peak_mem_mb"),
            "num_params": r.get("num_params"),
            "status": r.get("status"),
            "judge_reason": r.get("judge_reason"),
            "lesson": r.get("lesson"),
        } for r in records],
    }

    DOCS_DIR.mkdir(exist_ok=True)
    with open(OUTPUT_PATH, "w") as f:
        json.dump(data, f, ensure_ascii=False)
    print(f"[export] Wrote {OUTPUT_PATH} ({len(records)} experiments)")


def push():
    try:
        subprocess.run(
            ["git", "add", "docs/data.json"],
            cwd=PROJECT_DIR, check=True, capture_output=True,
        )
        subprocess.run(
            ["git", "commit", "-m", "Update dashboard data"],
            cwd=PROJECT_DIR, check=True, capture_output=True,
        )
        subprocess.run(
            ["git", "push"],
            cwd=PROJECT_DIR, check=True, capture_output=True,
        )
        print("[export] Pushed to GitHub")
    except subprocess.CalledProcessError as e:
        print(f"[export] Git push failed: {e.stderr.decode() if e.stderr else e}")


if __name__ == "__main__":
    export()
    if "--push" in sys.argv:
        push()
