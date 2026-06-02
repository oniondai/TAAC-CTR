"""
TAAC2026 Baseline 推理脚本（v2 增强版）
- 自动检测 v1/v2 模型 (通过 state_dict keys 区分)
- 复用 dataset.py / model_v2.py
- 严格按 BestDIN/PolyRec infer.py 的环境变量约定
- 输出 predictions.json: {"predictions": {user_id: prob}}

环境变量:
    MODEL_OUTPUT_PATH  必填: ckpt 目录（包含 best_model.pth / best_model_v2.pth + vocab.json）
    EVAL_DATA_PATH     必填: 测试数据目录（*.parquet + 可选 schema.json）
    EVAL_RESULT_PATH   必填: 预测输出目录
"""
import os
import sys
import json
import glob
import logging
import numpy as np
import torch
from torch.utils.data import DataLoader

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "model"))
# 把 src/ 加到 path 以便 from paths import ...
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from dataset import TAACOfficialDataset, collate_fn, DENSE_FIXED_LEN
from model import TAACBaseline
from model_v2 import TAACv2
from paths import VOCAB_PATH, INFO_PATH as _INFO_PATH

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


# ============== 工具函数 ==============
def find_checkpoint(model_dir):
    """找 ckpt 文件, 优先 v2"""
    candidates = [
        os.path.join(model_dir, "best_model_v2.pth"),
        os.path.join(model_dir, "best_model.pth"),
        os.path.join(model_dir, "model.pt"),
    ]
    for c in candidates:
        if os.path.exists(c):
            return c
    for root, _, files in os.walk(model_dir):
        for f in files:
            if f.endswith(".pt") or f.endswith(".pth"):
                return os.path.join(root, f)
    return None


def load_vocab_and_info(model_dir):
    """加载 vocab + info"""
    vocab_path = os.path.join(model_dir, "vocab.json")
    info_path = os.path.join(model_dir, "info.json")

    if not os.path.exists(vocab_path):
        vocab_path = VOCAB_PATH
    if not os.path.exists(info_path):
        info_path = _INFO_PATH

    vocab = json.load(open(vocab_path))
    info = None
    if os.path.exists(info_path):
        info = json.load(open(info_path))
    return vocab, info


def build_info_from_parquet(parquet_files):
    """从 parquet 文件反推 info（keys 分组）"""
    import pyarrow.parquet as pq
    schema = pq.read_schema(parquet_files[0])
    keys = [f.name for f in schema]

    user_int_keys = sorted(
        [k for k in keys if k.startswith("user_int_")],
        key=lambda x: int(x.split("_")[-1]),
    )
    item_int_keys = sorted(
        [k for k in keys if k.startswith("item_int_")],
        key=lambda x: int(x.split("_")[-1]),
    )
    user_dense_keys = sorted(
        [k for k in keys if k.startswith("user_dense_")],
        key=lambda x: int(x.split("_")[-1]),
    )
    seq_keys = [k for k in keys if "_seq_" in k]

    return {
        "user_int_keys": user_int_keys,
        "item_int_keys": item_int_keys,
        "user_dense_keys": user_dense_keys,
        "seq_keys": seq_keys,
        "dense_dim": len(user_dense_keys) * DENSE_FIXED_LEN,
    }


def load_test_parquet(data_dir):
    """加载测试 parquet 文件"""
    from datasets import load_dataset
    parquet_files = sorted(glob.glob(os.path.join(data_dir, "*.parquet")))
    if not parquet_files:
        if os.path.isfile(data_dir) and data_dir.endswith(".parquet"):
            parquet_files = [data_dir]
        else:
            raise FileNotFoundError(f"No parquet files in {data_dir}")
    logger.info(f"Found {len(parquet_files)} parquet files")
    ds = load_dataset("parquet", data_files=parquet_files)["train"]
    logger.info(f"Loaded {len(ds)} test rows, columns={len(ds.features)}")
    return ds, parquet_files


def detect_model_version(state_dict):
    """通过 state_dict keys 区分 v1 / v2"""
    keys = list(state_dict.keys())
    # v2 特有: user_id_hash.emb.weight, time_encoder, pair_encoder
    if any("user_id_hash" in k for k in keys):
        return "v2"
    if any("time_encoder" in k for k in keys):
        return "v2"
    return "v1"


# ============== 主流程 ==============
def main():
    model_dir = os.environ.get("MODEL_OUTPUT_PATH")
    data_dir = os.environ.get("EVAL_DATA_PATH")
    result_dir = os.environ.get("EVAL_RESULT_PATH")

    if not all([model_dir, data_dir, result_dir]):
        raise EnvironmentError(
            "Missing env vars. Need: MODEL_OUTPUT_PATH, EVAL_DATA_PATH, EVAL_RESULT_PATH"
        )

    os.makedirs(result_dir, exist_ok=True)
    device = "cpu"
    logger.info(f"MODEL_OUTPUT_PATH: {model_dir}")
    logger.info(f"EVAL_DATA_PATH: {data_dir}")
    logger.info(f"EVAL_RESULT_PATH: {result_dir}")
    logger.info(f"Device: {device}")

    # 1. 找 ckpt
    ckpt_path = find_checkpoint(model_dir)
    if ckpt_path is None:
        raise FileNotFoundError(f"No checkpoint found in {model_dir}")
    logger.info(f"Checkpoint: {ckpt_path}")

    # 2. 加载 state_dict 检测模型版本
    state_dict = torch.load(ckpt_path, map_location=device)
    model_version = detect_model_version(state_dict)
    logger.info(f"Detected model version: {model_version}")

    # 3. 加载 vocab / info
    vocab, info = load_vocab_and_info(model_dir)
    if info is None:
        _, parquet_files = load_test_parquet(data_dir)
        info = build_info_from_parquet(parquet_files)
    logger.info(f"Info: {len(info['user_int_keys'])} user_int, "
                f"{len(info['item_int_keys'])} item_int, "
                f"{len(info['seq_keys'])} seq, dense_dim={info['dense_dim']}")

    # 4. 加载测试数据
    ds, _ = load_test_parquet(data_dir)

    # 5. 构建 Dataset / DataLoader
    test_ds = TAACOfficialDataset(
        ds, vocab,
        info["user_int_keys"], info["item_int_keys"],
        info["user_dense_keys"], info["seq_keys"],
    )
    test_loader = DataLoader(
        test_ds, batch_size=64, shuffle=False,
        collate_fn=collate_fn, num_workers=0,
    )

    # 6. 构建对应版本模型
    if model_version == "v2":
        model = TAACv2(vocab, info, emb_dim=16, hidden_dims=(128, 64), dropout=0.3)
    else:
        model = TAACBaseline(vocab, info, emb_dim=16, hidden_dims=(128, 64), dropout=0.3)

    try:
        model.load_state_dict(state_dict)
    except RuntimeError as e:
        logger.error(f"State dict 不匹配: {e}")
        raise
    model.eval()
    model.to(device)
    logger.info(f"Model loaded: {sum(p.numel() for p in model.parameters()):,} params")

    # 7. 推理
    all_probs = []
    all_user_ids = []
    all_item_ids = []
    with torch.no_grad():
        for batch_idx, batch in enumerate(test_loader):
            logits = model(batch).squeeze(-1)
            probs = torch.sigmoid(logits).numpy()
            all_probs.extend(probs.tolist())
            # 用 raw ID (平台对账用)
            if "user_id_raw" in batch:
                all_user_ids.extend(batch["user_id_raw"].tolist())
                all_item_ids.extend(batch["item_id_raw"].tolist())
            else:
                all_user_ids.extend(batch["user_id"].tolist())
                all_item_ids.extend(batch["item_id"].tolist())
            if (batch_idx + 1) % 5 == 0:
                logger.info(f"  Batch {batch_idx+1}/{len(test_loader)}")

    logger.info(f"Inference done: {len(all_probs)} predictions")

    # 8. 写 predictions.json
    pred_dict = {}
    for uid, prob in zip(all_user_ids, all_probs):
        pred_dict.setdefault(int(uid), []).append(float(prob))
    pred_dict = {uid: float(np.mean(v)) for uid, v in pred_dict.items()}

    predictions = {"predictions": pred_dict}
    out_path = os.path.join(result_dir, "predictions.json")
    with open(out_path, "w") as f:
        json.dump(predictions, f)
    logger.info(f"Saved {len(pred_dict)} unique-user predictions to {out_path}")

    # 详细版
    detailed = [
        {"user_id": int(u), "item_id": int(i), "score": float(p)}
        for u, i, p in zip(all_user_ids, all_item_ids, all_probs)
    ]
    detail_path = os.path.join(result_dir, "predictions_detailed.json")
    with open(detail_path, "w") as f:
        json.dump(detailed, f, indent=2)
    logger.info(f"Detailed (with item_id) saved to {detail_path}")


if __name__ == "__main__":
    main()
