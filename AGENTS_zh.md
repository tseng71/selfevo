# SelfEvo - AGENTS.md

## 项目概述

SelfEvo 是一个运行在 Apple Silicon iMac 上的本地自我进化实验系统，唯一优化对象为 `mutable_train.py`。

## 项目结构

```
selfevo/
├── prepare.py            # 数据准备（TinyStories）
├── mutable_train.py      # 唯一优化对象：微型 Transformer 训练脚本
├── policy.py             # 策略模块：生成 patch plan
├── runner.py             # 执行模块：应用 patch、运行训练、调用 judge
├── judge.py              # 裁决模块：keep / discard / crash
├── main.py               # 入口点：启动 dashboard + 实验循环
├── constitution.md       # 最高规则（不可修改）
├── requirements.txt      # Python 依赖
├── data/                 # 数据缓存（tokenizer、train.bin、val.bin）
├── baseline/             # 当前最佳版本
├── memory.jsonl          # 实验历史记录
├── state.json            # 运行状态（dashboard IPC）
└── dashboard/
    ├── app.py            # FastAPI 后端
    └── static/           # 前端文件（HTML/JS/CSS）
```

## 开发约束

1. **只修改 mutable_train.py**：系统的自动进化只作用于这一个文件
2. **不修改评估逻辑**：val_loss 的计算方式和验证数据必须冻结
3. **Apple Silicon 兼容**：使用 PyTorch MPS 后端，不依赖 CUDA
4. **本地运行**：不做公网部署，不需要账户系统
5. **记录完整**：每轮实验都必须写入 memory.jsonl
6. **文档同步**：新增、删除或重命名模块/文件时，必须同步更新 README.md 和本文件的项目结构部分。修改运行依赖或启动步骤时，必须同步更新 README.md 的 Quick Start 部分。

## 关键流程

1. `prepare.py` 准备数据
2. `runner.py` 调用 `policy.py` 生成 patch plan
3. `runner.py` 应用 patch 到 `mutable_train.py`
4. 子进程运行训练
5. `judge.py` 比较结果
6. keep → 更新 baseline；discard/crash → 回滚
7. 写入 memory.jsonl
8. dashboard 展示结果

## 技术栈

- Python 3.10+
- PyTorch (MPS backend)
- FastAPI + uvicorn (dashboard)
- Chart.js (前端图表)
- tokenizers (BPE 分词)
- datasets (HuggingFace 数据加载)
