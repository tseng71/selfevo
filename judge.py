"""
SelfEvo - Judge Module
Compares experiment results against baseline, outputs keep/discard/crash verdict.
"""

import json
import math


# Threshold: new val_loss must be at least this much better to count as "keep"
IMPROVEMENT_THRESHOLD = 0.001
# If val_loss is within this range and params are fewer, keep (simplification win)
TIE_THRESHOLD = 0.02


def judge(new_result: dict, baseline_result: dict) -> dict:
    """
    Compare new experiment result against baseline.

    Returns:
        dict with keys:
            verdict: "keep" | "discard" | "crash"
            reason: human-readable explanation
    """
    # --- Crash detection ---
    status = new_result.get("status", "error")
    if status != "ok":
        failure_map = {
            "nan": "Training produced NaN values",
            "oom": "Out of memory during training",
            "error": "Runtime error during training",
            "timeout": "Training exceeded time limit",
            "syntax": "Syntax error in training script",
        }
        reason = failure_map.get(status, f"Training failed with status: {status}")
        return {"verdict": "crash", "reason": reason}

    new_loss = new_result.get("val_loss")
    if new_loss is None or (isinstance(new_loss, float) and (math.isnan(new_loss) or math.isinf(new_loss))):
        return {"verdict": "crash", "reason": "Invalid val_loss (None/NaN/Inf)"}

    # --- First experiment (no baseline) ---
    if baseline_result is None:
        return {"verdict": "keep", "reason": f"First experiment, establishing baseline (val_loss={new_loss:.4f})"}

    baseline_loss = baseline_result.get("val_loss")
    if baseline_loss is None:
        return {"verdict": "keep", "reason": f"No valid baseline loss, accepting new result (val_loss={new_loss:.4f})"}

    # --- Comparison ---
    improvement = baseline_loss - new_loss
    new_params = new_result.get("num_params", 0)
    base_params = baseline_result.get("num_params", 0)

    # Clear improvement
    if improvement > IMPROVEMENT_THRESHOLD:
        pct = (improvement / baseline_loss) * 100
        return {
            "verdict": "keep",
            "reason": f"val_loss improved: {baseline_loss:.4f} -> {new_loss:.4f} ({pct:.1f}% better)",
        }

    # Tie but simpler model
    if abs(improvement) <= TIE_THRESHOLD and new_params < base_params and base_params > 0:
        param_reduction = ((base_params - new_params) / base_params) * 100
        return {
            "verdict": "keep",
            "reason": f"Similar val_loss ({new_loss:.4f} vs {baseline_loss:.4f}) but {param_reduction:.1f}% fewer parameters",
        }

    # Regression or no improvement
    if improvement < -IMPROVEMENT_THRESHOLD:
        pct = (-improvement / baseline_loss) * 100
        return {
            "verdict": "discard",
            "reason": f"val_loss regressed: {baseline_loss:.4f} -> {new_loss:.4f} ({pct:.1f}% worse)",
        }

    # No meaningful change
    return {
        "verdict": "discard",
        "reason": f"No significant improvement: {baseline_loss:.4f} -> {new_loss:.4f} (delta={improvement:.4f})",
    }


if __name__ == "__main__":
    # Example usage
    baseline = {"val_loss": 4.5, "num_params": 2000000, "status": "ok"}
    new = {"val_loss": 4.3, "num_params": 2000000, "status": "ok"}
    result = judge(new, baseline)
    print(json.dumps(result, indent=2))
