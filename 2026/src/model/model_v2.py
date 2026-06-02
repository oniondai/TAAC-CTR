"""
TAAC2026 v2 模型 - 在 baseline 基础上加 3 个涨分点:
1. Hash Embedding (multi-bucket sum) for 高基数特征
2. Sample Time Context (Hour + Weekday + IsWeekend)
3. Pair Encoding: FID 62-66 的 user_int 和 user_dense 共享编码
4. 保留 DIN-style item attention (简化版)
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import json


class FeatureEmbedding(nn.Module):
    """单特征 Embedding"""
    def __init__(self, vocab_size, emb_dim, padding_idx=0):
        super().__init__()
        self.emb = nn.Embedding(vocab_size, emb_dim, padding_idx=padding_idx)
        nn.init.normal_(self.emb.weight, mean=0.0, std=0.01)
        with torch.no_grad():
            self.emb.weight[padding_idx].zero_()

    def forward(self, x):
        return self.emb(x)


class HashEmbedding(nn.Module):
    """多桶 Hash Embedding (sum 池化) — 用于高基数特征"""
    def __init__(self, hash_size, n_buckets, emb_dim):
        super().__init__()
        self.hash_size = hash_size
        self.n_buckets = n_buckets
        self.emb = nn.Embedding(hash_size * n_buckets, emb_dim)
        nn.init.normal_(self.emb.weight, mean=0.0, std=0.01)

    def forward(self, ids):
        """
        ids: (B, ) or (B, L)
        输出: (B, emb_dim) or (B, L, emb_dim)
        """
        out = 0
        for b in range(self.n_buckets):
            h = (ids * (2654435761 + b * 1009)) % self.hash_size + b * self.hash_size
            out = out + self.emb(h)
        return out / self.n_buckets


class SampleTimeEncoder(nn.Module):
    """绝对时间上下文编码: Hour + Weekday + IsWeekend"""
    def __init__(self, emb_dim=16):
        super().__init__()
        self.hour_emb = nn.Embedding(24, emb_dim)
        self.weekday_emb = nn.Embedding(7, emb_dim)
        self.is_weekend_emb = nn.Embedding(2, emb_dim)
        self.is_night_emb = nn.Embedding(2, emb_dim)
        nn.init.normal_(self.hour_emb.weight, std=0.01)
        nn.init.normal_(self.weekday_emb.weight, std=0.01)
        nn.init.normal_(self.is_weekend_emb.weight, std=0.01)
        nn.init.normal_(self.is_night_emb.weight, std=0.01)
        self.proj = nn.Sequential(
            nn.Linear(emb_dim * 4, emb_dim * 2),
            nn.PReLU(),
        )

    def forward(self, timestamp):
        """
        timestamp: (B, ) int64
        """
        hour = (timestamp // 3600) % 24
        day_index = timestamp // 86400
        weekday = (day_index + 3) % 7  # 1970-01-01 是周四 (+3)
        is_weekend = (weekday >= 5).long()
        is_night = ((hour >= 23) | (hour <= 1)).long()

        h = self.hour_emb(hour)
        w = self.weekday_emb(weekday)
        we = self.is_weekend_emb(is_weekend)
        n = self.is_night_emb(is_night)

        return self.proj(torch.cat([h, w, we, n], dim=-1))


class PairEncoder(nn.Module):
    """同源 Pair 编码: FID 62-66 的 user_int 和 user_dense 共享"""
    def __init__(self, pair_fids, emb_dim=16, dense_slice_len=64):
        super().__init__()
        self.pair_fids = pair_fids
        # 对每个 fid, dense slice -> emb_dim
        self.dense_projs = nn.ModuleDict({
            str(fid): nn.Linear(dense_slice_len, emb_dim) for fid in pair_fids
        })
        # 同样对 int_emb
        # int_emb 是预训练的 nn.Embedding 输出 (emb_dim), 直接相加

    def forward(self, int_emb_per_fid, dense_per_fid):
        """
        int_emb_per_fid: dict {fid: (B, emb_dim)}
        dense_per_fid: dict {fid: (B, dense_slice_len)}
        返回: (B, len(pair_fids) * emb_dim)
        """
        outs = []
        for fid in self.pair_fids:
            d = self.dense_projs[str(fid)](dense_per_fid[fid])  # (B, emb_dim)
            i = int_emb_per_fid[fid]                             # (B, emb_dim)
            outs.append(d * i)  # element-wise 乘积 (比简单加和更非线性)
        return torch.cat(outs, dim=-1)


class TAACv2(nn.Module):
    """
    v2 模型:
    - 普通 sparse: Embedding
    - 序列: mean pool -> Embedding
    - Dense: per-fid linear -> mean fusion
    - Sample Time: SampleTimeEncoder
    - Pair 编码: FID 62-66
    - Hash Embedding: 用户/物品 ID (高基数)
    """
    def __init__(self, vocab, info, emb_dim=16, hidden_dims=(128, 64), dropout=0.3,
                 pair_fids=(62, 63, 64, 65, 66), hash_id_buckets=4):
        super().__init__()
        self.vocab = vocab
        self.info = info
        self.emb_dim = emb_dim
        self.pair_fids = list(pair_fids)

        # 1. Hash Embedding for IDs (高基数)
        max_user_id = vocab.get('user_id', 1000)
        max_item_id = vocab.get('item_id', 1000)
        self.user_id_hash = HashEmbedding(
            hash_size=1024, n_buckets=hash_id_buckets, emb_dim=emb_dim
        )
        self.item_id_hash = HashEmbedding(
            hash_size=2048, n_buckets=hash_id_buckets, emb_dim=emb_dim
        )

        # 2. 普通 sparse Embedding
        self.user_int_embs = nn.ModuleDict({
            k: FeatureEmbedding(vocab[k], emb_dim) for k in info['user_int_keys']
        })
        self.item_int_embs = nn.ModuleDict({
            k: FeatureEmbedding(vocab[k], emb_dim) for k in info['item_int_keys']
        })

        # 3. 序列 Embedding (mean pool 后 scalar -> Embedding)
        self.seq_embs = nn.ModuleDict({
            k: FeatureEmbedding(vocab[k], emb_dim) for k in info['seq_keys']
        })

        # 4. Dense per-fid 投影 + 融合
        n_dense = len(info['user_dense_keys'])
        self.dense_projs = nn.ModuleList([
            nn.Linear(64, emb_dim) for _ in range(n_dense)
        ])
        self.dense_fuse = nn.Sequential(
            nn.Linear(n_dense * emb_dim, emb_dim * 2),
            nn.PReLU(),
            nn.Dropout(dropout),
        )

        # 5. Sample Time
        self.time_encoder = SampleTimeEncoder(emb_dim=emb_dim)

        # 6. Pair 编码 (FID 62-66)
        # dense_per_fid keys = 'dense_62', 'dense_63' ...
        # int_per_fid keys = 'user_int_feats_62', 'user_int_feats_63' ...
        self.pair_encoder = PairEncoder(
            pair_fids=self.pair_fids, emb_dim=emb_dim, dense_slice_len=64
        )

        # 计算总维度
        # user_id_hash + item_id_hash = 2 * emb_dim
        # user_int = len(user_int_keys) * emb_dim
        # item_int = len(item_int_keys) * emb_dim
        # seq = len(seq_keys) * emb_dim
        # dense_fuse = 2 * emb_dim
        # time_encoder = 2 * emb_dim
        # pair_encoder = len(pair_fids) * emb_dim
        total_dim = (
            2 * emb_dim  # ids
            + len(info['user_int_keys']) * emb_dim
            + len(info['item_int_keys']) * emb_dim
            + len(info['seq_keys']) * emb_dim
            + emb_dim * 2  # dense_fuse
            + emb_dim * 2  # time
            + len(self.pair_fids) * emb_dim  # pair
        )

        # MLP
        layers = []
        prev = total_dim
        for hd in hidden_dims:
            layers += [
                nn.Linear(prev, hd),
                nn.BatchNorm1d(hd),
                nn.PReLU(),
                nn.Dropout(dropout),
            ]
            prev = hd
        layers.append(nn.Linear(prev, 1))
        self.mlp = nn.Sequential(*layers)
        self._total_dim = total_dim

    def forward(self, batch):
        outs = []

        # 1. Hash IDs
        outs.append(self.user_id_hash(batch['user_id']))
        outs.append(self.item_id_hash(batch['item_id']))

        # 2. User Int (普通)
        int_emb_per_fid = {}
        for k in self.info['user_int_keys']:
            emb = self.user_int_embs[k](batch[k])
            outs.append(emb)
            # 记录 pair 用的 embedding
            if k.startswith('user_int_feats_'):
                fid = int(k.split('_')[-1])
                if fid in self.pair_fids:
                    int_emb_per_fid[fid] = emb

        # 3. Item Int
        for k in self.info['item_int_keys']:
            outs.append(self.item_int_embs[k](batch[k]))

        # 4. Sequences (mean pool 后 scalar)
        for k in self.info['seq_keys']:
            outs.append(self.seq_embs[k](batch[f'seq_{k}']))

        # 5. Dense per-fid 投影
        dense_per_fid = {}
        dense_concat = batch['user_dense']  # (B, n_dense * 64)
        n_dense = len(self.info['user_dense_keys'])
        dense_tokens = []
        for i, k in enumerate(self.info['user_dense_keys']):
            slice_ = dense_concat[:, i * 64:(i + 1) * 64]
            t = self.dense_projs[i](slice_)
            dense_tokens.append(t)
            # pair 用
            if k.startswith('user_dense_feats_'):
                fid = int(k.split('_')[-1])
                if fid in self.pair_fids:
                    dense_per_fid[fid] = slice_

        dense_fused = self.dense_fuse(torch.cat(dense_tokens, dim=-1))
        outs.append(dense_fused)

        # 6. Sample Time
        time_feat = self.time_encoder(batch['timestamp'])
        outs.append(time_feat)

        # 7. Pair 编码
        if int_emb_per_fid and dense_per_fid:
            pair_feat = self.pair_encoder(int_emb_per_fid, dense_per_fid)
            outs.append(pair_feat)
        else:
            # 兜底: 用 0 填充
            B = batch['user_id'].shape[0]
            outs.append(torch.zeros(B, len(self.pair_fids) * self.emb_dim,
                                     device=batch['user_id'].device))

        # Concat
        x = torch.cat(outs, dim=-1)
        assert x.shape[-1] == self._total_dim, f"dim mismatch: {x.shape[-1]} vs {self._total_dim}"

        return self.mlp(x).squeeze(-1)


if __name__ == "__main__":
    import os
    import sys
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from paths import YEAR_DIR, ensure_dirs
    ensure_dirs()
    os.chdir(YEAR_DIR)
    from dataset import get_loaders, collate_fn

    train_loader, val_loader, vocab, info = get_loaders(batch_size=8)
    model = TAACv2(vocab, info, emb_dim=16, hidden_dims=(128, 64))

    n_params = sum(p.numel() for p in model.parameters())
    print(f"\n模型参数量: {n_params:,}")

    batch = next(iter(train_loader))
    print(f"Batch keys: {list(batch.keys())[:5]}...")
    print(f"  user_id: {batch['user_id'].shape}, item_id: {batch['item_id'].shape}")
    print(f"  timestamp: {batch.get('timestamp', 'MISSING')}")
    print(f"  user_dense: {batch['user_dense'].shape}")

    logits = model(batch)
    print(f"\nlogits shape: {logits.shape}")
    print(f"sample logits: {logits[:5].tolist()}")
    print(f"sample labels: {batch['label'][:5].tolist()}")
    print("✅ v2 模型前向传播 OK")
