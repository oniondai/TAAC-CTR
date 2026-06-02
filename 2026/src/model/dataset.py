"""
TAAC2026 官方 demo 数据集加载器
- 自动处理稀疏(scalar+list<int>)/稠密(list<float>)/序列(list<int>)特征
- 自动扫描 vocab 并保存到 vocab.json
- 支持 DataLoader collate_fn（变长序列 padding）
"""
import os
import json
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from datasets import load_dataset

# 仓库内统一路径 (相对本文件位置的 2026/, paths.py 在 src/)
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from paths import LOCAL_PARQUET, VOCAB_PATH

# 序列最大长度（CPU 训练用短一些）
MAX_SEQ_LEN = 64
# Dense 特征固定长度（变长的用 mean pool + pad/truncate 到此长度）
DENSE_FIXED_LEN = 64
# Embedding 维度
EMB_DIM = 16
# 词汇表上限（超过此值的视为长尾，hash 桶处理）
VOCAB_CAP = 10000


def hash_bucket(val, cap):
    """把超过 vocab_cap 的值映射到 cap 以内"""
    return (val * 2654435761) % cap


def build_vocab(ds, user_int_keys, item_int_keys, seq_keys):
    """扫描数据集，构建 vocab 大小字典"""
    vocab = {}

    # IDs
    vocab['user_id'] = len(set(ds['user_id'])) + 1
    vocab['item_id'] = len(set(ds['item_id'])) + 1

    # User/Item 稀疏特征
    for k in user_int_keys + item_int_keys:
        col = ds[k]
        sample = col[0]
        if isinstance(sample, list):
            flat = []
            for row in col:
                if row is not None:
                    for v in row:
                        if v is not None and v != 0:
                            flat.append(v)
            uv = set(flat)
        else:
            uv = set(v for v in col if v is not None)
        vocab[k] = min(len(uv) + 1, VOCAB_CAP + 1)

    # 序列特征：每个序列单独 vocab
    for k in seq_keys:
        vals = set()
        for row in ds[k]:
            if row:
                for v in row:
                    if v != 0:
                        vals.add(v)
        vocab[k] = min(len(vals) + 1, VOCAB_CAP + 1)

    return vocab


class TAACOfficialDataset(Dataset):
    """官方 demo 数据集 Dataset"""

    def __init__(self, ds, vocab, user_int_keys, item_int_keys, user_dense_keys,
                 seq_keys, max_seq_len=MAX_SEQ_LEN, dense_fixed_len=DENSE_FIXED_LEN):
        self.ds = ds
        self.vocab = vocab
        self.user_int_keys = user_int_keys
        self.item_int_keys = item_int_keys
        self.user_dense_keys = user_dense_keys
        self.seq_keys = seq_keys
        self.max_seq_len = max_seq_len
        self.dense_fixed_len = dense_fixed_len

    def __len__(self):
        return len(self.ds)

    def _safe_idx(self, k, v):
        """把值映射到 vocab 范围内"""
        if v is None or v == 0:
            return 0  # padding
        vocab_size = self.vocab.get(k, VOCAB_CAP + 1)
        if v >= vocab_size:
            # cap 至少为 1，避免除零
            cap = max(1, vocab_size - 1)
            v = hash_bucket(v, cap) + 1
        # 二次保险：clamp
        return max(0, min(v, vocab_size - 1))

    def _process_list_int(self, k, lst):
        """处理 list<int> 稀疏特征 -> 单个 int（取 mean of hashes）"""
        if lst is None or len(lst) == 0:
            return 0
        vals = [v for v in lst if v is not None and v != 0]
        if not vals:
            return 0
        # 用 hash bucket 处理每个值后取 mean
        vocab_size = self.vocab.get(k, VOCAB_CAP + 1)
        bucketed = [self._safe_idx(k, v) for v in vals]
        return int(np.mean(bucketed))

    def _process_dense(self, lst):
        """处理 list<float> -> 固定长度 tensor"""
        if lst is None or len(lst) == 0:
            return np.zeros(self.dense_fixed_len, dtype=np.float32)
        arr = np.array(lst, dtype=np.float32)
        if len(arr) >= self.dense_fixed_len:
            return arr[:self.dense_fixed_len]
        # 填充
        padded = np.zeros(self.dense_fixed_len, dtype=np.float32)
        padded[:len(arr)] = arr
        return padded

    def _process_seq(self, lst):
        """处理序列 -> (padded_seq, length)"""
        if lst is None or len(lst) == 0:
            return np.zeros(self.max_seq_len, dtype=np.int64), 0
        # 过滤 0
        vals = [v for v in lst if v != 0]
        if not vals:
            return np.zeros(self.max_seq_len, dtype=np.int64), 0
        # 截断到尾部 max_seq_len
        vals = vals[-self.max_seq_len:]
        # hash bucket
        vocab_size = self.vocab.get(self._seq_key, VOCAB_CAP + 1)
        seq = [self._safe_idx(self._seq_key, v) for v in vals]
        # padding
        padded = np.zeros(self.max_seq_len, dtype=np.int64)
        padded[:len(seq)] = seq
        return padded, len(seq)

    def __getitem__(self, idx):
        row = self.ds[idx]
        sample = {}

        # IDs
        sample['user_id'] = self._safe_idx('user_id', row['user_id'])
        sample['item_id'] = self._safe_idx('item_id', row['item_id'])
        # 保留原始 ID (供平台对账使用)
        sample['user_id_raw'] = int(row['user_id']) if row['user_id'] is not None else 0
        sample['item_id_raw'] = int(row['item_id']) if row['item_id'] is not None else 0

        # User Int 特征
        for k in self.user_int_keys:
            v = row[k]
            if isinstance(v, list):
                sample[k] = self._process_list_int(k, v)
            else:
                sample[k] = self._safe_idx(k, v)

        # Item Int 特征
        for k in self.item_int_keys:
            v = row[k]
            if isinstance(v, list):
                sample[k] = self._process_list_int(k, v)
            else:
                sample[k] = self._safe_idx(k, v)

        # User Dense 特征（拼接所有）
        dense_concat = []
        for k in self.user_dense_keys:
            dense_concat.append(self._process_dense(row[k]))
        sample['user_dense'] = np.concatenate(dense_concat)  # shape=(n_dense * DENSE_FIXED_LEN,)

        # 序列特征（mean pool 后转 int 占位） - 用 sum/len 转成单值
        for k in self.seq_keys:
            self._seq_key = k
            vals = row[k]
            if vals:
                nonzero = [v for v in vals if v != 0]
                if nonzero:
                    vocab_size = self.vocab.get(k, VOCAB_CAP + 1)
                    bucketed = [self._safe_idx(k, v) for v in nonzero]
                    # mean pool + clamp 确保 < vocab_size
                    mean_val = int(np.mean(bucketed))
                    sample[f'seq_{k}'] = max(0, min(mean_val, vocab_size - 1))
                else:
                    sample[f'seq_{k}'] = 0
            else:
                sample[f'seq_{k}'] = 0

        # Labels
        sample['label'] = float(row['label_type'] - 1)  # 1->0, 2->1
        sample['timestamp'] = int(row['timestamp']) if row['timestamp'] is not None else 0

        return sample


def collate_fn(batch):
    """把 dict 列表转成 batch tensor"""
    out = {}
    # ID/scalar 特征
    for key in ['user_id', 'item_id', 'label', 'timestamp']:
        dtype = torch.long if key in ('user_id', 'item_id', 'timestamp') else torch.float32
        out[key] = torch.tensor([b[key] for b in batch], dtype=dtype)
    # Raw ID (平台对账)
    if 'user_id_raw' in batch[0]:
        out['user_id_raw'] = torch.tensor([b['user_id_raw'] for b in batch], dtype=torch.long)
        out['item_id_raw'] = torch.tensor([b['item_id_raw'] for b in batch], dtype=torch.long)
    # User/Item Int 特征
    for k in batch[0].keys():
        if k.startswith('user_int_feats_') or k.startswith('item_int_feats_'):
            out[k] = torch.tensor([b[k] for b in batch], dtype=torch.long)
    # 序列（mean pool 后是 scalar）
    for k in batch[0].keys():
        if k.startswith('seq_'):
            out[k] = torch.tensor([b[k] for b in batch], dtype=torch.long)
    # Dense
    out['user_dense'] = torch.tensor(np.stack([b['user_dense'] for b in batch]), dtype=torch.float32)
    return out


def get_loaders(batch_size=64, train_ratio=0.8, seed=42):
    """获取 train/val DataLoader"""
    ds = load_dataset("parquet", data_files=LOCAL_PARQUET)['train']

    keys = list(ds.features.keys())
    user_int_keys = sorted([k for k in keys if k.startswith('user_int_')],
                            key=lambda x: int(x.split('_')[-1]))
    item_int_keys = sorted([k for k in keys if k.startswith('item_int_')],
                            key=lambda x: int(x.split('_')[-1]))
    user_dense_keys = sorted([k for k in keys if k.startswith('user_dense_')],
                              key=lambda x: int(x.split('_')[-1]))
    seq_keys = [k for k in keys if '_seq_' in k]

    # Vocab：尝试从 cache 读
    if os.path.exists(VOCAB_PATH):
        vocab = json.load(open(VOCAB_PATH))
        print(f"✅ 从 {VOCAB_PATH} 加载 vocab")
    else:
        print("🚀 构建 vocab...")
        vocab = build_vocab(ds, user_int_keys, item_int_keys, seq_keys)
        with open(VOCAB_PATH, 'w') as f:
            json.dump(vocab, f, indent=2)
        print(f"✅ vocab 已保存到 {VOCAB_PATH}")

    # 划分 train/val
    n = len(ds)
    n_train = int(n * train_ratio)
    indices = list(range(n))
    import random
    random.seed(seed)
    random.shuffle(indices)
    train_indices = indices[:n_train]
    val_indices = indices[n_train:]

    train_ds = TAACOfficialDataset(ds.select(train_indices), vocab, user_int_keys, item_int_keys,
                                    user_dense_keys, seq_keys)
    val_ds = TAACOfficialDataset(ds.select(val_indices), vocab, user_int_keys, item_int_keys,
                                  user_dense_keys, seq_keys)

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True,
                              collate_fn=collate_fn, num_workers=0)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False,
                            collate_fn=collate_fn, num_workers=0)

    return train_loader, val_loader, vocab, {
        'user_int_keys': user_int_keys,
        'item_int_keys': item_int_keys,
        'user_dense_keys': user_dense_keys,
        'seq_keys': seq_keys,
        'dense_dim': len(user_dense_keys) * DENSE_FIXED_LEN,
    }


if __name__ == "__main__":
    train_loader, val_loader, vocab, info = get_loaders()
    print(f"\n=== Vocab 概览 ===")
    print(f"  user_id: {vocab['user_id']}")
    print(f"  item_id: {vocab['item_id']}")
    print(f"  user_int_feats 总数: {len(info['user_int_keys'])}")
    print(f"  item_int_feats 总数: {len(info['item_int_keys'])}")
    print(f"  user_dense_feats: {len(info['user_dense_keys'])} x {DENSE_FIXED_LEN} = {info['dense_dim']}")
    print(f"  序列特征: {len(info['seq_keys'])}")

    print(f"\n=== 加载测试 ===")
    print(f"  train batches: {len(train_loader)}, val batches: {len(val_loader)}")
    batch = next(iter(train_loader))
    print(f"  batch keys ({len(batch)}): {list(batch.keys())[:5]}...")
    print(f"  user_id shape: {batch['user_id'].shape}")
    print(f"  user_dense shape: {batch['user_dense'].shape}")
    print(f"  label shape: {batch['label'].shape}, sample: {batch['label'][:5]}")
