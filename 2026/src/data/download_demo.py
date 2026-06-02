"""
把 HF 上的 TAAC2026 公开 demo 数据集 (1000 行) 下载到本地 parquet
- 训练时 dataset.py 直接读这个 parquet，无需联网
- 代理设置: 优先用环境变量 HTTP_PROXY / HTTPS_PROXY;
           若未设且能连 huggingface.co, 直连;
           若你在中国大陆, 需要自己 export HTTP_PROXY=http://127.0.0.1:8118 之类

输出: YEAR_DIR/data_official.parquet  (相对本仓库的 2026/ 目录)
"""
import os
import sys

# 让 paths 可被找到 (paths.py 在 src/ 下, 本文件在 src/data/)
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from paths import LOCAL_PARQUET, YEAR_DIR, ensure_dirs

from datasets import load_dataset

ensure_dirs()

print("🌐 正在从 HuggingFace 下载 TAAC2026/data_sample_1000 ...")
ds = load_dataset("TAAC2026/data_sample_1000")['train']
ds.to_parquet(LOCAL_PARQUET)
print(f"✅ 已保存到 {LOCAL_PARQUET}")
print(f"   大小: {os.path.getsize(LOCAL_PARQUET)/1024/1024:.1f} MB")
print(f"   行数: {len(ds)}")
print(f"   字段: {list(ds.features.keys())[:5]} ... (共 {len(ds.features)} 个)")
