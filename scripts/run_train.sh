#!/bin/bash
# TiBERT 继续预训练启动脚本
# 用法: bash scripts/run_train.sh

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"
VENV="$PROJECT_ROOT/.venv/bin/python"

echo "============================================================"
echo "TiBERT 继续预训练"
echo "============================================================"
echo "项目目录: $PROJECT_ROOT"
echo "Python:   $VENV"
echo "开始时间: $(date)"

nohup $VENV "$SCRIPT_DIR/continued_pretrain.py" \
  --max_samples 2000000 \
  --epochs 3 \
  --batch_size 32 \
  --gradient_accumulation_steps 1 \
  --max_length 256 \
  --learning_rate 5e-5 \
  --warmup_steps 500 \
  --save_steps 5000 \
  --logging_steps 500 \
  --eval_before_train \
  > "$SCRIPT_DIR/train.log" 2>&1 &

echo "进程 PID: $!"
echo "日志文件: $SCRIPT_DIR/train.log"
echo "============================================================"
