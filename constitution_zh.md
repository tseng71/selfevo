# SelfEvo Constitution - 最高规则

本文件定义了 SelfEvo 系统的不可违反约束。任何模块都不得修改本文件内容。

## 1. 唯一优化对象

系统只允许修改 `mutable_train.py`。不得修改其他模块的核心逻辑。

## 2. 不可修改项

以下内容在任何实验中都不得被修改：

- **验证数据集**：`data/val.bin` 的内容和加载方式
- **评估逻辑**：`evaluate()` 函数和 `val_loss` 的计算方式
- **输出 Schema**：训练脚本的 JSON 输出字段和格式
- **预算上限**：`max_steps` 不得超过 PRD 定义的预算范围
- **裁决逻辑**：`judge.py` 的评判规则
- **记忆格式**：`memory.jsonl` 的记录结构

## 3. 允许修改项

在 `mutable_train.py` 中，以下区域可以被 patch 修改：

- 模型配置（层数、宽度、头数、dropout）
- 优化器参数（学习率、权重衰减、betas）
- 学习率调度（warmup、decay 策略）
- 批量大小和梯度累积
- 模型架构细节（MLP ratio、norm 选择等）

## 4. 安全约束

- crash 不得污染 baseline
- 每轮实验必须完整记录到 memory.jsonl
- baseline 只在 keep 时更新
- 系统崩溃后必须能恢复
- 用户随时可以暂停和回滚

## 5. 规则优先级

规则 > 策略 > 实验结果

系统不得通过修改评估标准来"作弊"获得更好的指标。
