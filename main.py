"""
SelfEvo - Main Entry Point
Starts the dashboard server and experiment loop.
"""

import argparse
import json
import os
import signal
import sys
import threading
import time
from datetime import datetime
from pathlib import Path

PROJECT_DIR = Path(__file__).parent

# Load .env file if it exists
_env_file = PROJECT_DIR / ".env"
if _env_file.exists():
    for line in _env_file.read_text().splitlines():
        line = line.strip()
        if line and not line.startswith("#") and "=" in line:
            key, _, value = line.partition("=")
            os.environ.setdefault(key.strip(), value.strip())
STATE_PATH = PROJECT_DIR / "state.json"
MEMORY_PATH = PROJECT_DIR / "memory.jsonl"
BASELINE_DIR = PROJECT_DIR / "baseline"
DATA_DIR = PROJECT_DIR / "data"

# Add project to path
sys.path.insert(0, str(PROJECT_DIR))


def load_state():
    if not STATE_PATH.exists():
        return {}
    try:
        return json.loads(STATE_PATH.read_text())
    except (json.JSONDecodeError, FileNotFoundError):
        return {}


def save_state(state):
    STATE_PATH.write_text(json.dumps(state, indent=2))


def check_data_ready():
    """Check if data has been prepared."""
    train_bin = DATA_DIR / "train.bin"
    val_bin = DATA_DIR / "val.bin"
    return train_bin.exists() and val_bin.exists()


def prepare_data():
    """Run data preparation."""
    print("[main] Data not found. Running prepare.py...")
    from prepare import main as prepare_main
    prepare_main()


def init_baseline():
    """Initialize baseline if needed."""
    from runner import init_baseline as runner_init_baseline
    runner_init_baseline()


def experiment_loop():
    """Run the experiment loop in a background thread."""
    from runner import run_single_experiment

    print("[main] Experiment loop started.")

    while True:
        state = load_state()

        # Check for commands
        command = state.get("command")
        status = state.get("status", "idle")

        if command == "pause" or status == "paused":
            state["status"] = "paused"
            state["command"] = None
            save_state(state)
            time.sleep(1)
            continue

        if command == "step":
            # Run single experiment
            state["command"] = None
            state["status"] = "running"
            save_state(state)
            print("\n[main] Running single experiment...")
            try:
                record = run_single_experiment()
                print(f"  Result: {record.get('status')} | val_loss: {record.get('val_loss')}")
                print(f"  Reason: {record.get('judge_reason')}")
            except Exception as e:
                print(f"  Error: {e}")
                state = load_state()
                state["status"] = "error"
                state["error"] = str(e)
                save_state(state)
            state = load_state()
            state["status"] = "idle"
            save_state(state)
            continue

        if command == "start" or status in ("running", "training"):
            state["command"] = None
            state["status"] = "running"
            save_state(state)

            print("\n[main] Running next experiment...")
            try:
                record = run_single_experiment()
                print(f"  Result: {record.get('status')} | val_loss: {record.get('val_loss')}")
                print(f"  Reason: {record.get('judge_reason')}")
            except Exception as e:
                print(f"  Error: {e}")
                state = load_state()
                state["status"] = "error"
                state["error"] = str(e)
                save_state(state)
                time.sleep(5)
                continue

            # Brief pause between experiments
            time.sleep(2)
            continue

        # Idle - wait for commands
        time.sleep(1)


def start_dashboard(host="0.0.0.0", port=8000):
    """Start the FastAPI dashboard server."""
    import uvicorn
    sys.path.insert(0, str(PROJECT_DIR / "dashboard"))
    from dashboard.app import app

    print(f"[main] Dashboard starting at http://localhost:{port}")
    uvicorn.run(app, host=host, port=port, log_level="warning")


def main():
    parser = argparse.ArgumentParser(description="SelfEvo - Self-Evolving Tiny LLM System")
    parser.add_argument("--port", type=int, default=8000, help="Dashboard port (default: 8000)")
    parser.add_argument("--host", default="0.0.0.0", help="Dashboard host (default: 0.0.0.0)")
    parser.add_argument("--dashboard-only", action="store_true", help="Only start dashboard, no experiment loop")
    parser.add_argument("--no-dashboard", action="store_true", help="Only run experiments, no dashboard")
    parser.add_argument("--auto-start", action="store_true", help="Auto-start experiment loop on launch")
    args = parser.parse_args()

    print("=" * 50)
    print("  SelfEvo - Self-Evolving Tiny LLM System")
    print("=" * 50)

    # Step 1: Check and prepare data
    if not check_data_ready():
        prepare_data()

    if not check_data_ready():
        print("[main] ERROR: Data preparation failed. Please run 'python prepare.py' manually.")
        sys.exit(1)

    print("[main] Data: OK")

    # Step 2: Initialize baseline
    init_baseline()
    print("[main] Baseline: OK")

    # Step 3: Initialize state
    state = load_state()
    if args.auto_start:
        state["status"] = "running"
        state["command"] = "start"
    else:
        state["status"] = "idle"
    state["last_update"] = datetime.now().isoformat()
    save_state(state)

    # Step 4: Start components
    if args.no_dashboard:
        # Just run experiment loop in main thread
        experiment_loop()
    elif args.dashboard_only:
        start_dashboard(args.host, args.port)
    else:
        # Start experiment loop in background thread
        exp_thread = threading.Thread(target=experiment_loop, daemon=True)
        exp_thread.start()
        print("[main] Experiment loop: Running in background")

        # Start dashboard in main thread (blocks)
        start_dashboard(args.host, args.port)


if __name__ == "__main__":
    main()
