"""
SelfEvo - Policy Module (v2: AI-Powered)
Uses AI (Gemini / OpenAI / Claude) to analyze experiment history and generate
intelligent code modifications. Falls back to heuristic rules if no API is available.

Supported providers (set via environment variable):
  - GEMINI_API_KEY  → Google Gemini
  - OPENAI_API_KEY  → OpenAI (GPT-4o / GPT-5.4 etc.)
  - ANTHROPIC_API_KEY → Anthropic Claude (Opus 4.6 etc.)

If multiple keys are set, priority: Gemini > OpenAI > Claude.
Override with AI_PROVIDER=gemini|openai|claude.
"""

import json
import os
import random
import re
import traceback
from pathlib import Path
from collections import Counter

MEMORY_PATH = Path(__file__).parent / "memory.jsonl"
BASELINE_SCRIPT = Path(__file__).parent / "baseline" / "mutable_train.py"
HISTORY_WINDOW = 20

EXPERIMENT_CLASSES = [
    "repair", "simplification", "architecture",
    "optimizer", "schedule", "batching", "exploration",
]


# ============================================================
# Utilities
# ============================================================

def read_baseline_config():
    """Parse the CONFIG section of the baseline script to get current values."""
    if not BASELINE_SCRIPT.exists():
        return {}
    code = BASELINE_SCRIPT.read_text()
    config = {}
    patterns = {
        "n_layer": r"^n_layer\s*=\s*(\d+)",
        "n_head": r"^n_head\s*=\s*(\d+)",
        "n_embd": r"^n_embd\s*=\s*(\d+)",
        "block_size": r"^block_size\s*=\s*(\d+)",
        "dropout": r"^dropout\s*=\s*([\d.]+)",
        "learning_rate": r"^learning_rate\s*=\s*([\de.\-+]+)",
        "weight_decay": r"^weight_decay\s*=\s*([\d.]+)",
        "beta1": r"^beta1\s*=\s*([\d.]+)",
        "beta2": r"^beta2\s*=\s*([\d.]+)",
        "grad_clip": r"^grad_clip\s*=\s*([\d.]+)",
        "batch_size": r"^batch_size\s*=\s*(\d+)",
        "gradient_accumulation_steps": r"^gradient_accumulation_steps\s*=\s*(\d+)",
        "warmup_steps": r"^warmup_steps\s*=\s*(\d+)",
        "max_steps": r"^max_steps\s*=\s*(\d+)",
    }
    for key, pattern in patterns.items():
        m = re.search(pattern, code, re.MULTILINE)
        if m:
            config[key] = m.group(1)
    return config


def load_history(n=HISTORY_WINDOW):
    if not MEMORY_PATH.exists():
        return []
    records = []
    with open(MEMORY_PATH) as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records[-n:]


def detect_phase(history):
    if not history:
        return "baseline"
    recent = history[-5:] if len(history) >= 5 else history
    statuses = [r.get("status", "discard") for r in recent]
    crashes = sum(1 for s in statuses if s == "crash")
    keeps = sum(1 for s in statuses if s == "keep")
    if crashes >= 3:
        return "repair"
    if keeps == 0 and len(recent) >= 5:
        return "exploration"
    if keeps >= 2:
        return "exploitation"
    return "exploration"


# ============================================================
# AI-Powered Policy (Gemini / OpenAI / Claude)
# ============================================================

# Provider config: (env_var, default_model)
AI_PROVIDERS = {
    "gemini":   ("GEMINI_API_KEY",    "gemini-2.5-pro"),
    "openai":   ("OPENAI_API_KEY",    "gpt-4.1"),
    "claude":   ("ANTHROPIC_API_KEY", "claude-opus-4-20250514"),
}

# Allow user to override model name: AI_MODEL=gpt-5.4 etc.
AI_MODEL_ENV = "AI_MODEL"
AI_PROVIDER_ENV = "AI_PROVIDER"
STATE_PATH = Path(__file__).parent / "state.json"


def _load_ai_settings_from_state():
    """Read AI provider/model override from state.json (set via dashboard)."""
    if not STATE_PATH.exists():
        return None, None
    try:
        with open(STATE_PATH) as f:
            state = json.load(f)
        return state.get("ai_provider"), state.get("ai_model")
    except Exception:
        return None, None


def _detect_provider():
    """Detect which AI provider to use. Priority: env var > state.json > auto-detect."""
    # 1. Explicit env var
    forced = os.environ.get(AI_PROVIDER_ENV, "").lower().strip()
    if forced and forced in AI_PROVIDERS:
        env_var, _ = AI_PROVIDERS[forced]
        if os.environ.get(env_var):
            return forced
        print(f"[policy] AI_PROVIDER={forced} but {env_var} not set, trying others...")

    # 2. Dashboard setting (state.json)
    state_provider, _ = _load_ai_settings_from_state()
    if state_provider and state_provider in AI_PROVIDERS:
        env_var, _ = AI_PROVIDERS[state_provider]
        if os.environ.get(env_var):
            return state_provider

    # 3. Auto-detect by priority
    for provider in ["gemini", "openai", "claude"]:
        env_var, _ = AI_PROVIDERS[provider]
        if os.environ.get(env_var):
            return provider
    return None


def _build_ai_prompt(baseline_code, cfg, history, phase):
    """Build the prompt for the AI agent."""

    # Summarize recent experiment history
    history_summary = []
    for r in history[-15:]:
        entry = {
            "patch": r.get("patch_summary", ""),
            "class": r.get("experiment_class", ""),
            "val_loss": r.get("val_loss"),
            "status": r.get("status"),
            "reason": r.get("judge_reason", ""),
            "hypothesis": r.get("hypothesis", ""),
            "params": r.get("num_params"),
            "time_sec": r.get("train_time_sec"),
        }
        history_summary.append(entry)

    # Find best val_loss
    best_vl = None
    for r in history:
        if r.get("status") == "keep" and r.get("val_loss") is not None:
            if best_vl is None or r["val_loss"] < best_vl:
                best_vl = r["val_loss"]

    # Count how many recent experiments were config-only vs code changes
    config_only_count = sum(
        1 for r in history[-20:]
        if r.get("patch_summary", "").count("->") == 1
        and " = " in r.get("patch_summary", "").split("->")[0]
    )

    prompt = f"""You are an expert AI researcher optimizing a tiny Transformer language model trained on TinyStories.

## Current Training Script (FULL)
```python
{_extract_mutable_sections(baseline_code)}
```

## Best val_loss achieved so far: {best_vl}
## Current phase: {phase}
## Total experiments: {len(history)} ({config_only_count} of last 20 were config-only tweaks)

## Recent Experiment History (last 15)
```json
{json.dumps(history_summary, indent=2)}
```

## STEP 1: ANALYZE ITERATION PATTERNS (required)

Before proposing your next experiment, you MUST analyze the experiment history above and identify:
1. **What worked**: Which types of modifications led to "keep" results? What do they have in common?
2. **What failed**: Which modifications were "discard" or "crash"? What patterns of failure do you see?
3. **Trends**: Is val_loss improving, plateauing, or getting worse? What direction should we explore next?
4. **Unexplored territory**: What promising modification types haven't been tried yet?
5. **Key insight**: Based on the above analysis, what is the single most promising direction for the next experiment?

Include your analysis in the "analysis" field of your JSON output.

## STEP 2: PROPOSE MODIFICATION

Based on your analysis, propose a modification to improve val_loss. You have FULL FREEDOM to change:
- **CONFIG values** (n_layer, n_head, n_embd, learning_rate, batch_size, dropout, etc.)
- **Model architecture** (attention, MLP, normalization, embeddings, init, etc.)
- **Training loop** (optimizer, LR schedule, gradient handling, loss function, etc.)
- **Data loading** (batching strategy, data augmentation, sequence packing, etc.)
- **Add new imports** if needed (e.g. from torch.optim.lr_scheduler import ...)

You can combine multiple types of changes in one experiment. Be creative and strategic.

IMPORTANT: Do NOT repeat modifications that already failed in the history above. Check the history carefully before proposing.

## Rules
- max_steps is FIXED at 500 (do NOT change it — this ensures fair comparison)
- The EVALUATION section and OUTPUT schema are IMMUTABLE (evaluation logic must stay identical)
- The find string must be EXACT text from the current script (including whitespace)
- Use \\n for newlines in your find/replace strings
- Make sure resulting code is valid Python
- Do NOT repeat failed experiments from the history
- You may add new code, new classes, new functions — no size limit on changes

## Output Format
Return ONLY valid JSON:
{{
    "analysis": {{
        "what_worked": "Summary of successful modifications and their common traits",
        "what_failed": "Summary of failed modifications and failure patterns",
        "trend": "Current optimization trend and momentum",
        "unexplored": "Promising directions not yet tried",
        "key_insight": "The single most important insight driving this experiment"
    }},
    "experiment_class": "architecture|optimizer|schedule|training",
    "hypothesis": "Why this specific code change will help, grounded in your analysis",
    "expected_effect": "Expected impact on val_loss",
    "mutations": [
        {{"find": "exact text from script", "replace": "replacement code"}}
    ]
}}
"""
    return prompt


def _extract_mutable_sections(code):
    """Return the full training script so Gemini can see the complete picture."""
    return code


def _call_gemini(api_key, model, prompt):
    """Call Gemini API and return response text."""
    from google import genai
    client = genai.Client(api_key=api_key)
    response = client.models.generate_content(
        model=model,
        contents=prompt,
        config={"temperature": 0.7, "max_output_tokens": 16384},
    )
    return response.text.strip()


def _call_openai(api_key, model, prompt):
    """Call OpenAI API and return response text."""
    from openai import OpenAI
    client = OpenAI(api_key=api_key)
    response = client.chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": prompt}],
        temperature=0.7,
        max_tokens=16384,
    )
    return response.choices[0].message.content.strip()


def _call_claude(api_key, model, prompt):
    """Call Anthropic Claude API and return response text."""
    import anthropic
    client = anthropic.Anthropic(api_key=api_key)
    response = client.messages.create(
        model=model,
        max_tokens=16384,
        messages=[{"role": "user", "content": prompt}],
        temperature=0.7,
    )
    return response.content[0].text.strip()


def generate_patch_plan_ai(history, cfg, phase):
    """Use AI (Gemini/OpenAI/Claude) to generate a patch plan."""
    provider = _detect_provider()
    if not provider:
        return None

    env_var, default_model = AI_PROVIDERS[provider]
    api_key = os.environ.get(env_var, "")

    # Model priority: env var > state.json > default
    model = os.environ.get(AI_MODEL_ENV, "").strip()
    if not model:
        _, state_model = _load_ai_settings_from_state()
        model = state_model or default_model

    try:
        baseline_code = BASELINE_SCRIPT.read_text() if BASELINE_SCRIPT.exists() else ""
        prompt = _build_ai_prompt(baseline_code, cfg, history, phase)

        # Call the appropriate provider
        if provider == "gemini":
            text = _call_gemini(api_key, model, prompt)
        elif provider == "openai":
            text = _call_openai(api_key, model, prompt)
        elif provider == "claude":
            text = _call_claude(api_key, model, prompt)
        else:
            return None

        print(f"[policy/ai] Provider: {provider} ({model})")

        # Extract JSON from response (handle markdown code blocks)
        if "```json" in text:
            text = text.split("```json")[1].split("```")[0].strip()
        elif "```" in text:
            text = text.split("```")[1].split("```")[0].strip()

        # Robust JSON parsing: fix common issues
        # Remove trailing commas before } or ]
        text = re.sub(r',\s*([}\]])', r'\1', text)

        # Fix unescaped newlines inside JSON strings:
        # Replace real newlines inside strings with \\n
        def _fix_json_newlines(raw):
            """Escape literal newlines that appear inside JSON string values."""
            fixed = []
            in_string = False
            i = 0
            while i < len(raw):
                ch = raw[i]
                if ch == '\\' and in_string and i + 1 < len(raw):
                    fixed.append(ch)
                    fixed.append(raw[i + 1])
                    i += 2
                    continue
                if ch == '"':
                    in_string = not in_string
                if ch == '\n' and in_string:
                    fixed.append('\\n')
                elif ch == '\t' and in_string:
                    fixed.append('\\t')
                else:
                    fixed.append(ch)
                i += 1
            return ''.join(fixed)

        try:
            plan = json.loads(text)
        except json.JSONDecodeError:
            text = _fix_json_newlines(text)
            plan = json.loads(text)

        # Validate the plan
        if not plan.get("mutations") or not isinstance(plan["mutations"], list):
            return None

        for m in plan["mutations"]:
            if "find" not in m or "replace" not in m:
                return None
            # Verify the find string exists in baseline
            if m["find"] and m["find"] not in baseline_code:
                print(f"[policy/ai] Warning: find string not in baseline: {m['find'][:50]}")
                return None

        # Determine risk level
        high_risk = any(
            any(m.get("find", "").strip().startswith(p) for p in ["n_layer", "n_embd", "block_size", "class "])
            for m in plan["mutations"]
        )

        # Log the AI analysis if present
        analysis = plan.get("analysis", {})
        if analysis:
            insight = analysis.get("key_insight", "")
            if insight:
                print(f"[policy/ai] Key insight: {insight[:100]}")

        return {
            "experiment_class": plan.get("experiment_class", "exploration"),
            "target_zone": "CONFIG",
            "max_lines_changed": sum(m["replace"].count("\n") + 1 for m in plan["mutations"]),
            "mutation_style": "ai_generated",
            "hypothesis": plan.get("hypothesis", "AI-generated experiment"),
            "expected_effect": plan.get("expected_effect", "Unknown"),
            "rollback_trigger": "val_loss > baseline + 0.5 or crash",
            "mutations": plan["mutations"],
            "high_risk": high_risk,
            "source": provider,
            "analysis": analysis,
        }

    except Exception as e:
        print(f"[policy/ai] {provider} API error: {e}")
        traceback.print_exc()
        return None


# ============================================================
# Heuristic Fallback (original logic)
# ============================================================

def make_mutations_architecture(cfg):
    mutations = []
    n_layer = int(cfg.get("n_layer", 4))
    n_embd = int(cfg.get("n_embd", 128))
    n_head = int(cfg.get("n_head", 4))
    dropout = float(cfg.get("dropout", 0.1))

    if n_layer < 8:
        mutations.append({"find": f"n_layer = {n_layer}", "replace": f"n_layer = {n_layer + 1}",
            "hypothesis": f"Deeper ({n_layer}->{n_layer+1})", "expected_effect": "More capacity"})
    if n_layer > 2:
        mutations.append({"find": f"n_layer = {n_layer}", "replace": f"n_layer = {n_layer - 1}",
            "hypothesis": f"Shallower ({n_layer}->{n_layer-1})", "expected_effect": "Faster per step"})
    for delta in [32, 64]:
        for sign in [1, -1]:
            new_embd = n_embd + sign * delta
            if 32 <= new_embd <= 384 and new_embd % n_head == 0:
                mutations.append({"find": f"n_embd = {n_embd}", "replace": f"n_embd = {new_embd}",
                    "hypothesis": f"{'Wider' if sign > 0 else 'Narrower'} ({n_embd}->{new_embd})",
                    "expected_effect": "Changed capacity"})
    for new_heads in [2, 4, 6, 8]:
        if new_heads != n_head and n_embd % new_heads == 0:
            mutations.append({"find": f"n_head = {n_head}", "replace": f"n_head = {new_heads}",
                "hypothesis": f"Heads {n_head}->{new_heads}", "expected_effect": "Changed attention"})
    for new_drop in [0.0, 0.02, 0.05, 0.1, 0.15, 0.2]:
        if abs(new_drop - dropout) > 0.01:
            mutations.append({"find": f"dropout = {cfg.get('dropout', '0.1')}", "replace": f"dropout = {new_drop}",
                "hypothesis": f"Dropout {dropout}->{new_drop}", "expected_effect": "Changed regularization"})
    return mutations


def make_mutations_optimizer(cfg):
    mutations = []
    lr = cfg.get("learning_rate", "3e-4")
    lr_float = float(lr)
    for mult, desc in [(0.3, "lower"), (0.5, "lower"), (2.0, "higher"), (3.0, "higher")]:
        new_lr = lr_float * mult
        if 1e-5 <= new_lr <= 5e-3:
            new_lr_str = f"{new_lr:.1e}" if new_lr < 0.001 else f"{new_lr}"
            mutations.append({"find": f"learning_rate = {lr}", "replace": f"learning_rate = {new_lr_str}",
                "hypothesis": f"LR {desc} ({lr}->{new_lr_str})", "expected_effect": "Changed convergence"})
    wd = cfg.get("weight_decay", "0.1")
    wd_float = float(wd)
    for new_wd in [0.001, 0.01, 0.05, 0.1, 0.2, 0.3]:
        if abs(new_wd - wd_float) > 0.005:
            mutations.append({"find": f"weight_decay = {wd}", "replace": f"weight_decay = {new_wd}",
                "hypothesis": f"WD {wd}->{new_wd}", "expected_effect": "Changed regularization"})
    b2 = cfg.get("beta2", "0.95")
    b2_float = float(b2)
    for new_b2 in [0.9, 0.95, 0.98, 0.99, 0.999]:
        if abs(new_b2 - b2_float) > 0.005:
            mutations.append({"find": f"beta2 = {b2}", "replace": f"beta2 = {new_b2}",
                "hypothesis": f"Beta2 {b2}->{new_b2}", "expected_effect": "Changed moment estimation"})
    gc = cfg.get("grad_clip", "1.0")
    gc_float = float(gc)
    for new_gc in [0.1, 0.3, 0.5, 1.0, 2.0, 5.0]:
        if abs(new_gc - gc_float) > 0.05:
            mutations.append({"find": f"grad_clip = {gc}", "replace": f"grad_clip = {new_gc}",
                "hypothesis": f"Grad clip {gc}->{new_gc}", "expected_effect": "Changed gradient control"})
    return mutations


def make_mutations_schedule(cfg):
    mutations = []
    warmup = int(cfg.get("warmup_steps", 50))
    for new_w in [10, 20, 30, 50, 75, 100, 150]:
        if new_w != warmup:
            mutations.append({"find": f"warmup_steps = {warmup}", "replace": f"warmup_steps = {new_w}",
                "hypothesis": f"Warmup {warmup}->{new_w}", "expected_effect": "Changed warmup"})
    return mutations


def make_mutations_batching(cfg):
    mutations = []
    bs = int(cfg.get("batch_size", 32))
    for new_bs in [8, 16, 32, 48, 64, 96, 128]:
        if new_bs != bs:
            mutations.append({"find": f"batch_size = {bs}", "replace": f"batch_size = {new_bs}",
                "hypothesis": f"Batch {bs}->{new_bs}", "expected_effect": "Changed gradient noise"})
    ga = int(cfg.get("gradient_accumulation_steps", 1))
    for new_ga in [1, 2, 4]:
        if new_ga != ga:
            mutations.append({"find": f"gradient_accumulation_steps = {ga}",
                "replace": f"gradient_accumulation_steps = {new_ga}",
                "hypothesis": f"Accum {ga}->{new_ga}", "expected_effect": "Changed effective batch"})
    return mutations


MUTATION_GENERATORS = {
    "architecture": make_mutations_architecture,
    "optimizer": make_mutations_optimizer,
    "schedule": make_mutations_schedule,
    "batching": make_mutations_batching,
    "simplification": make_mutations_architecture,
    "exploration": make_mutations_architecture,
}


def select_experiment_class(phase, history):
    if phase == "repair":
        return "repair"
    if phase == "baseline":
        return random.choice(["optimizer", "architecture", "schedule", "batching"])
    class_success = Counter()
    class_total = Counter()
    for r in history:
        cls = r.get("experiment_class", "unknown")
        class_total[cls] += 1
        if r.get("status") == "keep":
            class_success[cls] += 1
    if phase == "exploitation":
        best_classes = [(c, class_success.get(c, 0) / class_total[c])
                        for c in EXPERIMENT_CLASSES if c != "repair" and class_total.get(c, 0) > 0 and class_success.get(c, 0) > 0]
        if best_classes:
            best_classes.sort(key=lambda x: x[1], reverse=True)
            top = best_classes[:3]
            weights = [s for _, s in top]
            return random.choices([c for c, _ in top], weights=weights, k=1)[0]
    underexplored = [c for c in EXPERIMENT_CLASSES if c != "repair" and class_total.get(c, 0) < 3]
    if underexplored:
        return random.choice(underexplored)
    return random.choice([c for c in EXPERIMENT_CLASSES if c != "repair"])


def generate_patch_plan_heuristic(history, cfg, phase):
    """Heuristic-based fallback patch plan generation."""
    experiment_class = select_experiment_class(phase, history)

    if experiment_class == "repair":
        return {
            "experiment_class": "repair", "target_zone": "CONFIG",
            "max_lines_changed": 10, "mutation_style": "revert",
            "hypothesis": "Revert to last known-good configuration",
            "expected_effect": "Restore stability",
            "rollback_trigger": "any crash", "mutations": [], "is_revert": True,
        }

    generator = MUTATION_GENERATORS.get(experiment_class)
    candidates = generator(cfg) if generator else []
    if not candidates:
        for gen in MUTATION_GENERATORS.values():
            candidates.extend(gen(cfg))
    if not candidates:
        candidates = make_mutations_optimizer(cfg)

    recent_patches = {r.get("patch_summary", "") for r in history[-10:]}
    untried = [m for m in candidates if f"{m['find']} -> {m['replace']}" not in recent_patches]
    mutation = random.choice(untried) if untried else random.choice(candidates)

    high_risk = any(mutation.get("find", "").startswith(p) for p in ["n_layer", "n_embd", "block_size"])
    return {
        "experiment_class": experiment_class, "target_zone": "CONFIG",
        "max_lines_changed": 2, "mutation_style": "tweak",
        "hypothesis": mutation["hypothesis"],
        "expected_effect": mutation["expected_effect"],
        "rollback_trigger": "val_loss > baseline + 0.5 or crash",
        "mutations": [{"find": mutation["find"], "replace": mutation["replace"]}],
        "high_risk": high_risk, "source": "heuristic",
    }


# ============================================================
# Main Entry Point
# ============================================================

def generate_patch_plan(history=None):
    """Generate a patch plan. Uses AI if available, falls back to heuristics."""
    if history is None:
        history = load_history()

    cfg = read_baseline_config()
    phase = detect_phase(history)

    # Handle repair first (no need for AI)
    if phase == "repair":
        return generate_patch_plan_heuristic(history, cfg, phase)

    # Try AI-powered generation
    ai_plan = generate_patch_plan_ai(history, cfg, phase)
    if ai_plan:
        src = ai_plan.get("source", "ai")
        print(f"[policy] Using {src}: {ai_plan.get('hypothesis', '')[:60]}")
        return ai_plan

    # Fallback to heuristics
    plan = generate_patch_plan_heuristic(history, cfg, phase)
    print(f"[policy] Using heuristic: {plan.get('hypothesis', '')[:60]}")
    return plan


if __name__ == "__main__":
    plan = generate_patch_plan()
    print(json.dumps(plan, indent=2))
