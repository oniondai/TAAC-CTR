"""
统一的仓库内路径工具
所有脚本都从 TAAC-CTR/2026 这个根目录出发，不依赖机器绝对路径

约定:
    REPO_ROOT = TAAC-CTR  (项目根)
    YEAR_DIR  = TAAC-CTR/2026  (年度工作目录)
"""
import os

# 2026/ 的绝对路径 = 本文件所在目录的爷爷目录
_THIS_DIR = os.path.dirname(os.path.abspath(__file__))  # src/
YEAR_DIR = os.path.dirname(_THIS_DIR)                     # 2026/
REPO_ROOT = os.path.dirname(YEAR_DIR)                     # TAAC-CTR/

# 数据/产物固定位置 (相对 YEAR_DIR)
DATA_DIR        = os.path.join(YEAR_DIR, "data")
MODELS_DIR      = os.path.join(YEAR_DIR, "models")
BUNDLE_DIR      = os.path.join(YEAR_DIR, "bundle")
LOG_DIR         = os.path.join(YEAR_DIR, "logs")

# 关键文件
LOCAL_PARQUET   = os.path.join(YEAR_DIR, "data_official.parquet")
VOCAB_PATH      = os.path.join(YEAR_DIR, "vocab.json")
INFO_PATH       = os.path.join(YEAR_DIR, "info.json")
CKPT_PATH_V1    = os.path.join(MODELS_DIR, "best_model.pth")
CKPT_PATH_V2    = os.path.join(MODELS_DIR, "best_model_v2.pth")
BUNDLE_ZIP      = os.path.join(BUNDLE_DIR, "inference.zip")
PREDICTIONS_OUT = os.path.join(BUNDLE_DIR, "predictions.json")

# Mock 数据
MOCK_CSV = os.path.join(YEAR_DIR, "Tenrec_dataset", "Tenrec", "ctr_data_1M.csv")


def ensure_dirs():
    """训练/推理前保证所有目录存在"""
    for d in [DATA_DIR, MODELS_DIR, BUNDLE_DIR, LOG_DIR, os.path.dirname(MOCK_CSV)]:
        os.makedirs(d, exist_ok=True)
