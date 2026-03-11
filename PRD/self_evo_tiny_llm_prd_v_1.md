# SelfEvo Tiny LLM Self-Evolving System PRD v1.0

## 1. Document Information

**Product Name**: SelfEvo
**Version**: v1.0
**Document Type**: Product Requirements Document (PRD)
**Target Platform**: Personal computers (macOS / Windows / Linux) — no dedicated GPU required
**Target Users**: Non-programmers or light-programmers who want to observe and control a self-evolving training experiment through a visual interface

---

## 2. One-Line Definition

SelfEvo is a local self-evolving experiment system that runs on personal computers: it wraps a tiny text LLM training script `mutable_train.py`, automatically proposes modifications, runs fixed-budget training, compares validation results, keeps winning versions, rolls back regressions, and presents experiment progress, key metrics, version changes, and human intervention controls through a local visual dashboard.

---

## 3. Background and Problem Definition

### 3.1 Background

Inspired by ideas like nanochat and Andrej Karpathy's "autoresearch" concept, a clear direction for research automation has emerged:

- Narrow "AI doing research" down to a small, verifiable closed loop;
- Each round, the system only modifies one well-defined target;
- Run under a fixed budget;
- Compare results using fixed metrics;
- Keep improvements, discard regressions;
- Accumulate experiment memory to inform the next round's direction.

The value of such systems is not in inventing major new theories, but in automating the "run small experiments → get fast feedback → accumulate experience" mechanism.

### 3.2 Current Problems

Existing discussions about "self-evolving agents / multi-agent research" often get stuck on three issues before development even begins:

1. **No single optimization target**: It's unclear whether to optimize the model, prompts, router, evaluator, or the agent organization itself;
2. **No hard evaluation loop**: The system can endlessly propose ideas, but there's no fixed benchmark to prove whether changes are actually effective;
3. **No locally runnable scale**: Many proposals assume Linux + NVIDIA GPU and cannot be verified directly on personal computers.

### 3.3 This Project's Answer

This project's v1 makes a decisive commitment to one direction:

- **Sole optimization target**: `mutable_train.py`
- **Training task**: Tiny decoder-only Transformer on TinyStories
- **Evaluation goal**: Minimize `val_loss` under a fixed training budget (v1.1 may add `val_bpb`)
- **Presentation**: Must provide a local visual dashboard for non-programmers to observe and intervene

---

## 4. Product Goals

### 4.1 Core Goals

Build a minimal closed-loop system that runs locally on personal computers, capable of:

1. Automatically modifying the tiny training script;
2. Automatically running fixed-budget training;
3. Automatically comparing training results;
4. Automatically deciding keep / discard / rollback;
5. Automatically recording experiment history;
6. Automatically proposing the next experiment direction based on history;
7. Transparently presenting all key states to users through a visual dashboard.

### 4.2 Success Criteria

The success criteria for v1 is not "training a strong LLM", but rather:

- Being able to stably complete multiple experiment rounds locally;
- Forming at least a baseline with several improved/regressed versions;
- Being able to automatically make keep/rollback decisions;
- Users can understand what the system is doing through the dashboard without reading code;
- Users can intervene at critical points.

### 4.3 Non-Goals

v1 explicitly does not pursue:

- Large model training;
- Public multi-user product;
- Multi-dataset, multi-task generalization;
- Distributed training;
- Multi-agent asynchronous large-scale collaboration;
- Allowing the system to modify its own supreme rules.

---

## 5. Product Principles

### 5.1 Single Optimization Target Principle

v1 only optimizes `mutable_train.py`. Multiple primary optimization targets must not be defined simultaneously.

### 5.2 Fixed Evaluation Principle

The dataset, validation set, training budget, and evaluation logic must be fixed to ensure experiments are comparable.

### 5.3 Small Steps, Fast Iteration Principle

Prioritize short budgets, small models, and small changes so that personal computers can deliver sufficiently fast feedback.

### 5.4 Rollback Principle

Any failed or regressed experiment must be automatically rollable to the current baseline.

### 5.5 Visibility-First Principle

This project targets non-programmers; all key states, metrics, and decisions must be visible on the local dashboard.

### 5.6 Rules Over Strategy Principle

The system may modify the training script, but must not modify evaluation criteria, validation data, or supreme constraints.

---

## 6. Target Users and Usage Scenarios

### 6.1 Target Users

- Individual researchers wanting to verify whether a self-evolving training system is feasible;
- Non-programmers willing to observe system behavior through charts and dashboards;
- Developers wanting to build and maintain code repositories with AI coding assistants;
- People wanting to verify AI research automation prototypes on local devices.

### 6.2 Typical Usage Scenarios

#### Scenario A: User Starts the System

The user opens the local dashboard, clicks "Start Experiment", and the system loads the current baseline, reads historical memory, and automatically begins the next experiment round.

#### Scenario B: User Observes Whether the System Is Improving

On the overview page, the user sees the current best `val_loss`, today's experiment count, the last 10 round results, keep/discard/crash ratios and trend charts, enabling them to judge whether the system is actually improving.

#### Scenario C: User Intervenes in High-Risk Experiments

When the system plans a major change, the dashboard prompts "High-risk experiment pending approval", and the user can review the hypothesis and expected impact before approving or rejecting.

#### Scenario D: User Reverts to Best Version

When the system performs poorly across multiple rounds, the user clicks "Restore to Best Version" on the control page, and the system automatically reverts to the baseline.

---

## 7. Product Scope

### 7.1 v1 Must-Do Scope

1. TinyStories data preparation;
2. Tiny decoder-only Transformer training script;
3. Fixed-budget training and validation;
4. Patch plan generation;
5. Patch application;
6. Keep / discard / rollback decisions;
7. Experiment history logging;
8. Local visual dashboard;
9. Human control interface;
10. Compatible with personal computers (macOS, Windows, Linux).

### 7.2 v1 Out-of-Scope

1. Public deployment;
2. Complex account and permission systems;
3. SaaS multi-user collaboration;
4. Large-scale distributed training;
5. Multi-branch parallel experiment graphs;
6. Automatic evaluator modification;
7. Automatic supreme rules modification;
8. Mobile app.

---

## 8. Core Concept Definitions

### 8.1 `mutable_train.py`

The system's sole optimization target. A runnable, evaluable, patchable tiny LLM training script.

### 8.2 One Experiment Round

An experiment round is defined as:

> Applying one patch to `mutable_train.py`, running one training and validation session under a fixed budget, recording the results, and deciding whether to keep the change.

### 8.3 Baseline

The current known-best version and its corresponding metrics.

### 8.4 Patch Plan

A structured proposal for the next round's modifications, including at minimum: experiment class, target zone, change magnitude, hypothesis, expected effect, and rollback trigger conditions.

### 8.5 Keep / Discard / Crash

- **Keep**: This round's results are better than baseline, or approximately equal but with reduced complexity;
- **Discard**: Ran successfully but results were not better;
- **Crash**: Runtime anomaly, such as syntax errors, NaN, OOM, timeout, etc.

### 8.6 Memory

Experiment memory store for recording historical results, failure types, and lessons learned, providing the basis for next-round decisions.

---

## 9. System Architecture

The v1 system consists of the following core files:

- `prepare.py`: Prepares TinyStories data
- `mutable_train.py`: Sole optimization target
- `policy.py`: Generates next-round patch plans based on history
- `runner.py`: Applies patches, runs training, collects logs
- `judge.py`: Decides keep / discard / crash based on results
- `memory.jsonl`: Records experiment history
- `baseline/`: Stores the current best version
- `dashboard/`: Local visual dashboard
- `constitution.md`: Supreme rules
- `AGENTS.md`: Project constraints for AI coding assistants

### 9.1 Module Responsibilities

#### Data Preparation Module
Responsible for downloading, cleaning, splitting, and caching TinyStories data.

#### Training Target Module
Responsible for defining the model, training loop, validation logic, and standardized output.

#### Policy Module
Responsible for deciding what to change next, which zone to target, and how much to change, based on historical experiments.

#### Execution Module
Responsible for applying patches, running training, and parsing output.

#### Judging Module
Responsible for comparing results and producing keep / discard / crash decisions.

#### Memory Module
Responsible for recording history, readable by the policy module.

#### Visualization Module
Responsible for presenting key metrics, trends, experiment lists, diffs, and control interfaces to users.

---

## 10. Functional Requirements

## 10.1 Data Preparation Module (`prepare.py`)

### Goal

Enable TinyStories data to be prepared once locally and reused multiple times.

### Functional Requirements

1. Support first-time data download or reading existing local data;
2. Support tokenizer initialization or loading;
3. Split data into train / val;
4. Generate reusable cache;
5. Skip completed steps on repeated runs;
6. Output summary of data size, vocabulary size, context length, etc.

### Acceptance Criteria

- First run successfully generates cache;
- Subsequent runs do not re-prepare;
- Train / val split is stable;
- Local cache path is clearly identifiable.

---

## 10.2 Mutable Training Script Module (`mutable_train.py`)

### Goal

Provide a clearly structured, stably running micro LLM training script suitable for automated small-step modifications.

### Content Requirements

Must include:

1. Model configuration section;
2. Decoder-only Transformer definition;
3. Optimizer configuration;
4. Learning rate schedule;
5. Data loading;
6. Fixed-budget training loop;
7. Validation logic;
8. Result output.

### Allowed Optimizations

1. Model depth, width, number of heads;
2. Dropout, MLP ratio, normalization details;
3. Learning rate, weight decay, betas;
4. Warmup, decay strategy;
5. Batch size, accumulation steps;
6. Local training stability and efficiency logic.

### Forbidden Optimizations

1. Validation set;
2. Evaluation metric definition;
3. Budget ceiling;
4. Result output schema;
5. External judge logic.

### Standard Output Fields

- `val_loss`
- `train_time_sec`
- `total_time_sec`
- `peak_mem_mb`
- `num_steps`
- `num_params`
- `status`

### Acceptance Criteria

- Running the script standalone completes one training and validation cycle;
- Output fields are stable;
- Short-budget training completes on personal computers;
- Code structure is clear enough for subsequent patching.

---

## 10.3 Policy Module (`policy.py`)

### Goal

Based on historical results, decide how to modify `mutable_train.py` in the next round.

### v1 Responsibilities

1. Read the most recent N experiment records;
2. Determine the current phase: exploration, exploitation, repair, or stagnation;
3. Select experiment class;
4. Select target zone;
5. Decide change magnitude;
6. Output a structured patch plan.

### v1 Experiment Classes

- `repair`
- `simplification`
- `architecture`
- `optimizer`
- `schedule`
- `batching`
- `exploration`
- `exploitation`

### Patch Plan Fields

- `experiment_class`
- `target_zone`
- `max_lines_changed`
- `mutation_style`
- `hypothesis`
- `expected_effect`
- `rollback_trigger`

### v1 Implementation Requirements

- Primarily heuristic rules;
- May include simple action scoring;
- Output must be structured;
- Policy model self-training is not required.

### Acceptance Criteria

- Can output valid patch plans based on history;
- Prioritizes repair after consecutive crashes;
- Increases exploration weight after consecutive stagnation;
- Output can be consumed by runner.

---

## 10.4 Execution Module (`runner.py`)

### Goal

Strictly execute one experiment round, ensuring reliability of patch application, training execution, result collection, and version management.

### Functional Requirements

1. Receive patch plan;
2. Generate and apply patch to `mutable_train.py`;
3. Run fixed-budget training;
4. Capture output results;
5. Call judge for evaluation;
6. Update baseline on keep results;
7. Rollback on discard / crash results;
8. Write to memory.

### Exception Handling

Must handle:

- Syntax errors;
- Runtime errors;
- NaN;
- OOM;
- Timeout;
- Missing output fields.

### Acceptance Criteria

- A single experiment round can complete automatically;
- Failed experiments do not corrupt baseline;
- Output results are consistent with memory;
- Repeated runs are stable.

---

## 10.5 Judging Module (`judge.py`)

### Goal

Centrally compare current experiment results with baseline and output explainable keep/discard decisions.

### Functional Requirements

1. Compare `val_loss`;
2. Identify anomalous results as crash;
3. Support keep for approximately equal results with reduced complexity;
4. Output keep / discard / crash;
5. Output brief reason explanation.

### v1 Judging Rules

- Primary metric: lower `val_loss` is better;
- Secondary metrics: time, memory, parameter count, complexity;
- Anomalies take highest priority.

### Acceptance Criteria

- Result determination is stable;
- No "cheating" by modifying the evaluator;
- Can provide readable reasons for the dashboard.

---

## 10.6 Memory Module (`memory.jsonl`)

### Goal

Record system experiment history, supporting review, statistics, and policy decisions.

### Fields Per Record

- experiment_id
- timestamp
- parent_version
- patch_summary
- experiment_class
- target_zone
- val_loss
- train_time_sec
- peak_mem_mb
- status
- failure_type
- judge_reason
- lesson

### Acceptance Criteria

- Every experiment round has a record;
- Field structure is stable;
- Dashboard can read directly;
- Policy can read directly.

---

## 10.7 Local Visual Dashboard

### Goal

Allow non-programmers to understand whether the system is improving, what each experiment round did, why decisions were made to keep or discard, and when human intervention is needed — without reading code or terminal logs.

### Key Principles

- Information should be clear, not complex or flashy;
- Runs locally, no public deployment required;
- Key metrics visible at a glance;
- Human control of key actions is allowed.

### Page Requirements

#### A. Overview Page

Displays:

- Current running status (running / paused / error)
- Current best version number
- Current best `val_loss`
- Most recent experiment result
- Today's experiment count
- Keep / discard / crash statistics
- Current phase (baseline / exploration / exploitation / repair)

#### B. Experiment History Page

Table showing:

- Experiment number
- Time
- Patch summary
- Experiment class
- Target zone
- `val_loss`
- Training duration
- Peak memory
- Status
- Reason summary

Filtering support:

- By status
- By class
- By time

#### C. Metrics Trend Page

Charts showing:

- `val_loss` trend line
- Keep rate trend
- Crash rate trend
- Success rate by experiment class
- Average improvement by target zone

#### D. Version Comparison Page

Displays:

- Baseline vs. selected version metric comparison
- Patch summary comparison
- Core hyperparameter differences
- Judge reasoning

#### E. Control Page

Provides buttons:

- Start running
- Pause
- Single experiment round
- Restore to best version
- Lock current baseline
- Allow high-risk experiments toggle
- Allow major changes toggle

### Key Interactions

1. When the system is about to execute a high-risk patch, it may pop up "Pending human approval";
2. On crash, the dashboard must display the failure type and recent log summary;
3. On keep/discard, a brief natural language reason must be displayed;
4. Users can pause the system at any time.

### Acceptance Criteria

- Non-programmers can understand the main pages;
- Can determine from the overview page whether the system is improving;
- Can learn from the history page what each round did;
- Can complete critical human interventions on the control page.

---

## 11. Key Workflows

### 11.1 First Launch Flow

1. User opens the local dashboard;
2. System checks whether data is prepared;
3. If not prepared, guides execution of `prepare.py`;
4. System loads baseline;
5. Dashboard displays "Ready";
6. User clicks "Start Experiment".

### 11.2 Single Experiment Round Flow

1. `policy.py` reads history;
2. Generates patch plan;
3. `runner.py` applies patch;
4. Runs fixed-budget training;
5. `judge.py` makes judgment;
6. Updates baseline or rolls back;
7. Writes to memory;
8. Dashboard refreshes to display results.

### 11.3 High-Risk Human Approval Flow

1. Policy determines this round is high-risk;
2. Dashboard marks "Pending approval";
3. User reviews hypothesis and expected impact;
4. User approves or rejects;
5. System continues or cancels this round.

### 11.4 Restore Best Version Flow

1. User clicks "Restore to Best Version";
2. Runner restores baseline file;
3. Dashboard updates status;
4. System continues subsequent experiments from baseline.

---

## 12. Non-Functional Requirements

### 12.1 Performance

- v1 single experiment round duration should be within acceptable range;
- Local dashboard refresh should be smooth;
- Reading history should not noticeably lag.

### 12.2 Reliability

- Crashes must not corrupt baseline;
- Memory writes must be atomic;
- Must be recoverable after interruption.

### 12.3 Understandability

- All status names should be clear;
- Keep/discard reasons must be readable;
- Non-programmers can understand the system's main behavior through the dashboard.

### 12.4 Maintainability

- File responsibilities are clear;
- Logs and data structures are unified;
- Easy for AI coding assistants to iterate on.

### 12.5 Local Compatibility

- Compatible with macOS (Apple Silicon), Windows, and Linux;
- Supports local Python / PyTorch MPS / CUDA / CPU environments;
- Does not depend on platform-specific capabilities.

---

## 13. Metrics Framework

### 13.1 Product Metrics

- Number of successfully completed experiment rounds
- Keep ratio
- Crash ratio
- Success rate of user controls through dashboard
- Baseline refresh count

### 13.2 Research Metrics

- Current best `val_loss`
- Improvement magnitude over initial baseline
- Success rate by experiment class
- Average improvement by target zone
- Consecutive rounds without improvement

### 13.3 Experience Metrics

- Can the user understand the current status within 1 minute
- Can the user see the most recent experiment result within 3 clicks
- Can the user independently complete pause/resume/rollback on the dashboard

---

## 14. Risks and Constraints

### 14.1 Compute Risk

Personal computers can run tiny LLM benchmarks, but are not suitable for high-density, large-scale experiments. Therefore v1 must control model size and budget.

### 14.2 Evaluation Overfitting Risk

If the system modifies evaluation paths it shouldn't, it may create a "looks stronger but is actually cheating" problem. Therefore evaluation logic must be frozen.

### 14.3 Experiment Noise Risk

Short-budget training inherently has high noise, so judge rules must be cautious. Reproduction experiments may be added in v1.1.

### 14.4 User Understanding Risk

If dashboard information is unclear, non-programmers will still lose their sense of control. Therefore the information architecture must be designed around four clear layers: "status, results, reasons, controls".

---

## 15. Milestones

### Milestone 1: Basic Training Target Runnable

Deliverables:

- `prepare.py`
- `mutable_train.py`
- Locally complete one short-budget training run

### Milestone 2: Single Experiment Round Closed Loop

Deliverables:

- `policy.py`
- `runner.py`
- `judge.py`
- `memory.jsonl`
- Keep / discard / rollback working

### Milestone 3: Local Visual Dashboard Online

Deliverables:

- Overview page
- History page
- Trend page
- Comparison page
- Control page

### Milestone 4: v1 Usable

Deliverables:

- Multiple experiment rounds running stably
- Dashboard fully functional
- Human intervention working
- Baseline can be refreshed

---

## 16. Acceptance Criteria

v1 is considered complete when the following conditions are met:

1. TinyStories data can be successfully prepared locally;
2. `mutable_train.py` can complete short-budget training on personal computers;
3. The system can complete at least one automated patch + training + judging round;
4. Keep / discard / crash logic works;
5. Baseline mechanism works;
6. History/memory records are complete;
7. Dashboard can display key states, experiment lists, and trend charts;
8. Users can complete start, pause, single experiment, and rollback through the dashboard;
9. Users do not need to read code to understand the system's general progress.

---

## 17. Post-v1 Iteration Directions

### v1.1

- Add `val_bpb`
- Add reproduction experiments
- Add more granular failure types
- Add clearer patch diff display

### v1.2

- Add multi-candidate patch comparison
- Add stronger action scoring
- Add "strategy suggestions" rather than automatic strategy modification

### v2

- Add branch experiments
- Add adopt mechanism
- Add stronger human review and annotation capabilities

---

## 18. Development Implementation Notes for AI Coding Assistants

### Overall Development Goal

Implement a locally running self-evolving experiment system with `mutable_train.py` as the sole optimization target. The system uses TinyStories as its dataset, trains a tiny decoder-only Transformer, compares `val_loss` under a fixed budget, and presents experiment progress and control interfaces to non-programmers through a local visual dashboard.

### Hard Constraints

1. May only optimize `mutable_train.py`;
2. Must not modify evaluation logic and validation data;
3. Must support keep / discard / rollback;
4. Must produce `memory.jsonl`;
5. Must provide a local visual dashboard;
6. Must be compatible with personal computers (macOS, Windows, Linux);
7. No public multi-user system.

### Deliverable Files

- `prepare.py`
- `mutable_train.py`
- `policy.py`
- `runner.py`
- `judge.py`
- `memory.jsonl`
- `dashboard/*`
- `constitution.md`
- `AGENTS.md`

---

## 19. Conclusion

SelfEvo v1 is not a "generalized AGI self-evolving platform", but a **clear, focused, and implementable** local research system:

- It optimizes only one target: `mutable_train.py`;
- It performs only one task: TinyStories tiny LLM short-budget training;
- It chases only one primary metric: `val_loss`;
- It must be visible, understandable, and controllable for non-programmers.

The purpose of this PRD is to move the project from conceptual discussion into a clearly developable state.
