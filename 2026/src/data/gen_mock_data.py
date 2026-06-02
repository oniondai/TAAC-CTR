"""
生成符合 TAAC2026-CTR-Baseline schema 的 mock 数据
- 字段: user_id, item_id, click, gender, age, video_category, hist_1 ~ hist_10
- 数据量: 10,000 行（CPU 训练用，速度合理）
- 输出: YEAR_DIR/Tenrec_dataset/Tenrec/ctr_data_1M.csv
"""
import csv
import random
import os
import sys

# 让 paths 可被找到 (paths.py 在 src/ 下, 本文件在 src/data/)
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from paths import MOCK_CSV, ensure_dirs

N_SAMPLES = 10000
N_USERS = 1000
N_ITEMS = 5000
N_CATEGORIES = 100

ensure_dirs()
os.makedirs(os.path.dirname(MOCK_CSV), exist_ok=True)

print(f"🚀 开始生成 {N_SAMPLES} 条 mock 数据到 {MOCK_CSV}")

with open(MOCK_CSV, "w", newline="", encoding="utf-8") as f:
    writer = csv.writer(f)
    writer.writerow([
        "user_id", "item_id", "click", "gender", "age", "video_category",
        "hist_1", "hist_2", "hist_3", "hist_4", "hist_5",
        "hist_6", "hist_7", "hist_8", "hist_9", "hist_10"
    ])

    for _ in range(N_SAMPLES):
        user_id = random.randint(1, N_USERS)
        item_id = random.randint(1, N_ITEMS)
        click = random.choices([0, 1], weights=[0.8, 0.2])[0]
        gender = random.choices([1, 2], weights=[0.55, 0.45])[0]
        age = random.choice([1, 2, 3, 4, 5, 6, 7])
        video_category = random.randint(0, N_CATEGORIES - 1)
        # 历史行为序列：10 个历史 item_id（0 表示没有）
        hist = [random.randint(1, N_ITEMS) for _ in range(10)]

        writer.writerow([user_id, item_id, click, gender, age, video_category] + hist)

print(f"✅ 数据生成完成: {MOCK_CSV}")
print(f"   文件大小: {os.path.getsize(MOCK_CSV) / 1024:.1f} KB")
