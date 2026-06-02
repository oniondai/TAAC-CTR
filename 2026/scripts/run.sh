#!/usr/bin/env bash
# TAAC2026 一键跑通脚本
#
# 用法:
#   bash scripts/run.sh all           # 完整流程: 下载数据 → 训练 → 推理 → 打包
#   bash scripts/run.sh data          # 只下载 demo 数据
#   bash scripts/run.sh train         # 只训练 (v2 模型)
#   bash scripts/run.sh train v1      # 训练 v1 (baseline) 模型
#   bash scripts/run.sh infer         # 用现有模型推理 (需要先准备 EVAL_DATA_PATH)
#   bash scripts/run.sh bundle        # 打包 infer bundle 给平台上传
#   bash scripts/run.sh clean         # 清理所有产物 (vocab/models/bundle)
#
# 环境要求:
#   - Python 3.8+
#   - pip install -r 2026/requirements.txt
#   - 如果在中国大陆, 需 export HTTP_PROXY=http://127.0.0.1:8118 后再跑
set -e

# ===== 0. 路径 =====
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
YEAR_DIR="$(dirname "$SCRIPT_DIR")"
REPO_ROOT="$(dirname "$YEAR_DIR")"
SRC_DIR="$YEAR_DIR/src"
DATA_DIR="$YEAR_DIR/data"
MODELS_DIR="$YEAR_DIR/models"
BUNDLE_DIR="$YEAR_DIR/bundle"
LOG_DIR="$YEAR_DIR/logs"

mkdir -p "$DATA_DIR" "$MODELS_DIR" "$BUNDLE_DIR" "$LOG_DIR"

# ===== 1. Python 解释器 =====
# 默认用本机 python3, 优先用项目级的 venv
if [ -d "$REPO_ROOT/../TAAC2026-CTR-Baseline/taac_venv" ]; then
    PY="$REPO_ROOT/../TAAC2026-CTR-Baseline/taac_venv/bin/python"
elif [ -d "$REPO_ROOT/taac_venv" ]; then
    PY="$REPO_ROOT/taac_venv/bin/python"
elif command -v python3 >/dev/null 2>&1; then
    PY=python3
else
    echo "❌ 找不到 python3, 请先装 Python 3.8+"
    exit 1
fi

# ===== 2. 切到 2026/ 工作目录 =====
cd "$YEAR_DIR"

# ===== 3. 工具函数 =====
log() { echo -e "\n\033[1;36m▶ $*\033[0m"; }
ok()  { echo -e "\033[1;32m✅ $*\033[0m"; }
err() { echo -e "\033[1;31m❌ $*\033[0m"; }

step_data() {
    log "[1/4] 下载 TAAC2026 官方 demo 数据 (HuggingFace TAAC2026/data_sample_1000)"
    if [ -f "data_official.parquet" ]; then
        ok "data_official.parquet 已存在, 跳过下载"
        return 0
    fi
    "$PY" "$SRC_DIR/data/download_demo.py"
    ok "数据下载完成: $(ls -lh data_official.parquet | awk '{print $5}')"
}

step_train() {
    local VERSION="${1:-v2}"
    log "[2/4] 训练 $VERSION 模型"
    if [ "$VERSION" = "v1" ]; then
        "$PY" "$SRC_DIR/model/main.py" 2>&1 | tee "$LOG_DIR/train_v1.log"
        CKPT="$MODELS_DIR/best_model.pth"
    else
        "$PY" "$SRC_DIR/model/main_v2.py" 2>&1 | tee "$LOG_DIR/train_v2.log"
        CKPT="$MODELS_DIR/best_model_v2.pth"
    fi
    if [ ! -f "$CKPT" ]; then
        err "训练后未生成 $CKPT, 看 $LOG_DIR/train_$VERSION.log"
        exit 1
    fi
    ok "训练完成, ckpt: $CKPT ($(du -h "$CKPT" | awk '{print $1}'))"
}

step_infer() {
    log "[3/4] 本地推理 (拿训练集前 20% 当测试, 验证全流程)"
    if [ ! -f "$MODELS_DIR/best_model_v2.pth" ] && [ ! -f "$MODELS_DIR/best_model.pth" ]; then
        err "没有可用的模型权重, 请先跑 train 步骤"
        exit 1
    fi

    # 准备 fake test
    TEST_DIR="/tmp/taac_fake_test_$$"
    mkdir -p "$TEST_DIR"
    "$PY" -c "
import pyarrow.parquet as pq, random, os
random.seed(42)
tbl = pq.read_table('data_official.parquet')
n = tbl.num_rows
idx = list(range(n)); random.shuffle(idx)
test_idx = idx[:int(n*0.2)]
pq.write_table(tbl.take(test_idx), '$TEST_DIR/test_part1.parquet')
print(f'  fake test rows: {len(test_idx)}')
"

    # 跑 infer
    MODEL_OUTPUT_PATH="$MODELS_DIR" \
    EVAL_DATA_PATH="$TEST_DIR" \
    EVAL_RESULT_PATH="$BUNDLE_DIR" \
    "$PY" "$SRC_DIR/infer/infer.py" 2>&1 | tee "$LOG_DIR/infer.log"

    rm -rf "$TEST_DIR"
    ok "推理完成, 预测: $BUNDLE_DIR/predictions.json"
}

step_bundle() {
    log "[4/4] 打包平台上传 bundle (zip + manifest)"
    "$PY" "$SRC_DIR/infer/make_bundle.py" 2>&1 | tee "$LOG_DIR/bundle.log"
}

step_clean() {
    log "🧹 清理产物..."
    rm -rf "$MODELS_DIR"/*.pth "$MODELS_DIR"/*.pt
    rm -f "$YEAR_DIR"/vocab.json "$YEAR_DIR"/info.json
    rm -rf "$BUNDLE_DIR"/* "$LOG_DIR"/*
    rm -f "$YEAR_DIR"/data_official.parquet
    rm -rf "$YEAR_DIR"/Tenrec_dataset
    ok "清理完成"
}

# ===== 4. 路由 =====
CMD="${1:-all}"
SUB="${2:-v2}"

case "$CMD" in
    data)    step_data ;;
    train)   step_train "$SUB" ;;
    infer)   step_infer ;;
    bundle)  step_bundle ;;
    clean)   step_clean ;;
    all)
        step_data
        step_train v2
        step_infer
        step_bundle
        ok "🎉 全流程跑完! 平台上传包在: $BUNDLE_DIR/"
        ;;
    *)
        echo "用法: $0 {all|data|train [v1|v2]|infer|bundle|clean}"
        exit 1
        ;;
esac
