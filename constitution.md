# SelfEvo Constitution - Supreme Rules

This file defines the inviolable constraints of the SelfEvo system. No module may modify the contents of this file.

## 1. Sole Optimization Target

The system is only allowed to modify `mutable_train.py`. No other module's core logic may be modified.

## 2. Immutable Items

The following must never be modified in any experiment:

- **Validation dataset**: The contents and loading method of `data/val.bin`
- **Evaluation logic**: The `evaluate()` function and `val_loss` computation
- **Output schema**: The JSON output fields and format of the training script
- **Budget ceiling**: `max_steps` must not exceed the budget range defined in the PRD
- **Judging logic**: The judging rules in `judge.py`
- **Memory format**: The record structure of `memory.jsonl`

## 3. Modifiable Items

Within `mutable_train.py`, the following areas may be modified by patches:

- Model configuration (layers, width, heads, dropout)
- Optimizer parameters (learning rate, weight decay, betas)
- Learning rate schedule (warmup, decay strategy)
- Batch size and gradient accumulation
- Model architecture details (MLP ratio, normalization choice, etc.)

## 4. Safety Constraints

- A crash must not corrupt the baseline
- Every experiment round must be fully recorded in memory.jsonl
- The baseline is only updated on a keep decision
- The system must be recoverable after a crash
- The user can pause and rollback at any time

## 5. Rule Priority

Rules > Strategy > Experiment Results

The system must not "cheat" by modifying evaluation criteria to achieve better metrics.
