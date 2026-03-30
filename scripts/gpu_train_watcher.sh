#!/bin/bash
# gpu_train_watcher.sh
# 每30分钟检测GPU空闲，有足够显存则自动启动训练
# Usage: 启动后会在后台持续运行，直到训练开始或手动停止

LOG="$HOME/.claude/gpu_train_watcher.log"
LOCK="$HOME/.claude/gpu_train_watcher.lock"
MODEL_CHECKPOINT="$HOME/.claude/gpu_training_active"

# GPU内存阈值（MB），需要至少这个空闲才启动训练
# POS分类器：DataParallel 模式，每卡约需 8GB（batch分布在多卡）
MIN_FREE_MB=8000

log() {
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] $1" | tee -a "$LOG"
}

already_running() {
    if [ -f "$LOCK" ]; then
        pid=$(cat "$LOCK" 2>/dev/null)
        if [ -n "$pid" ] && kill -0 "$pid" 2>/dev/null; then
            return 0
        fi
        rm -f "$LOCK"
    fi
    return 1
}

get_gpu_free() {
    # 返回GPU0空闲显存（MB）
    nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits -i 0 2>/dev/null | tr -d ' '
}

get_gpu_used() {
    nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits -i 0 2>/dev/null | tr -d ' '
}

already_training() {
    # 检查训练是否已在运行
    pgrep -f "train_pos_classifier\|finetune_qwen" > /dev/null 2>&1
}

start_pos_training() {
    log "GPU空闲，开始启动 POS 分类器训练..."
    cd /home/chenhao/Tibert-Classical
    .venv/bin/python scripts/train_pos_classifier.py >> "$HOME/.claude/pos_train.log" 2>&1 &
    TRAIN_PID=$!
    echo $TRAIN_PID > "$MODEL_CHECKPOINT"
    log "POS训练已启动 (PID=$TRAIN_PID)"
}

start_qwen_finetune() {
    log "GPU空闲，开始启动 Qwen 微调训练..."
    cd /home/chenhao/Tibert-Classical
    .venv/bin/python scripts/finetune_qwen_pos.py --train \
        --data_file data/corpus/grammar_analysis_train_50000.jsonl \
        >> "$HOME/.claude/qwen_train.log" 2>&1 &
    TRAIN_PID=$!
    echo $TRAIN_PID > "$MODEL_CHECKPOINT"
    log "Qwen微调已启动 (PID=$TRAIN_PID)"
}

# ── Main ─────────────────────────────────────────────────────────────────────

mkdir -p "$(dirname "$LOG")"

if already_running; then
    log "监控已在运行 (PID=$(cat $LOCK))，退出"
    exit 0
fi

echo $$ > "$LOCK"
log "GPU监控启动（每30分钟检查，最小空闲显存=${MIN_FREE_MB}MB）"

while true; do
    if already_training; then
        log "训练已在进行中，跳过..."
    else
        free_mb=$(get_gpu_free)
        used_mb=$(get_gpu_used)
        log "检测GPU: 已用=${used_mb}MB, 空闲=${free_mb}MB"

        if [ -n "$free_mb" ] && [ "$free_mb" -ge "$MIN_FREE_MB" ]; then
            log "GPU空闲（${free_mb}MB >= ${MIN_FREE_MB}MB），启动训练！"

            # 训练 POS 分类器
            start_pos_training

            rm -f "$LOCK"
            log "监控结束，训练已启动"
            exit 0
        else
            log "GPU被占用（${free_mb}MB < ${MIN_FREE_MB}MB），30分钟后重试"
        fi
    fi

    sleep 1800   # 30分钟
done
