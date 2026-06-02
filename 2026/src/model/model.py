"""
TAAC2026 官方 demo Baseline 模型
- 自动适配任意 vocab 字典
- Embedding for sparse features
- Linear projection for dense features
- Mean pool for sequences (handled in dataset)
- MLP + 1 个 logit head（二分类 label_type）
"""
import torch
import torch.nn as nn
import json
import math


class FeatureEmbedding(nn.Module):
    """单特征 Embedding 层"""
    def __init__(self, vocab_size, emb_dim):
        super().__init__()
        self.emb = nn.Embedding(vocab_size, emb_dim, padding_idx=0)
        nn.init.normal_(self.emb.weight, mean=0.0, std=0.01)
        # padding_idx=0 不更新
        with torch.no_grad():
            self.emb.weight[0].zero_()

    def forward(self, x):
        return self.emb(x)


class TAACBaseline(nn.Module):
    """多特征融合 MLP"""
    def __init__(self, vocab, info, emb_dim=16, hidden_dims=(128, 64), dropout=0.3):
        super().__init__()
        self.vocab = vocab
        self.info = info
        self.emb_dim = emb_dim

        # 1. Embedding for IDs
        self.user_id_emb = FeatureEmbedding(vocab['user_id'], emb_dim)
        self.item_id_emb = FeatureEmbedding(vocab['item_id'], emb_dim)

        # 2. Embedding for user/item int features（共用 dim，因为已经是分散的）
        self.user_int_embs = nn.ModuleDict({
            k: FeatureEmbedding(vocab[k], emb_dim) for k in info['user_int_keys']
        })
        self.item_int_embs = nn.ModuleDict({
            k: FeatureEmbedding(vocab[k], emb_dim) for k in info['item_int_keys']
        })

        # 3. Embedding for sequences (mean-pooled scalars)
        self.seq_embs = nn.ModuleDict({
            k: FeatureEmbedding(vocab[k], emb_dim) for k in info['seq_keys']
        })

        # 4. Dense 特征：直接 linear projection
        dense_input_dim = info['dense_dim']  # 10 * 64 = 640
        self.dense_proj = nn.Sequential(
            nn.Linear(dense_input_dim, emb_dim * 4),
            nn.PReLU(),
            nn.Dropout(dropout),
        )

        # 计算总特征维度
        # user_id + item_id = 2
        # user_int = len(user_int_keys)  (each -> emb_dim)
        # item_int = len(item_int_keys)
        # seq = len(seq_keys)
        # dense_proj = emb_dim * 4
        total_emb_dim = (2 + len(info['user_int_keys']) + len(info['item_int_keys']) + len(info['seq_keys'])) * emb_dim
        total_emb_dim += emb_dim * 4  # dense projection output

        # MLP head
        layers = []
        prev_dim = total_emb_dim
        for hd in hidden_dims:
            layers += [
                nn.Linear(prev_dim, hd),
                nn.BatchNorm1d(hd),
                nn.PReLU(),
                nn.Dropout(dropout),
            ]
            prev_dim = hd
        layers.append(nn.Linear(prev_dim, 1))
        self.mlp = nn.Sequential(*layers)

        self._total_dim = total_emb_dim

    def forward(self, batch):
        embs = []

        # IDs
        embs.append(self.user_id_emb(batch['user_id']))
        embs.append(self.item_id_emb(batch['item_id']))

        # User Int
        for k in self.info['user_int_keys']:
            embs.append(self.user_int_embs[k](batch[k]))

        # Item Int
        for k in self.info['item_int_keys']:
            embs.append(self.item_int_embs[k](batch[k]))

        # Sequences (mean-pooled)
        for k in self.info['seq_keys']:
            embs.append(self.seq_embs[k](batch[f'seq_{k}']))

        # Dense
        embs.append(self.dense_proj(batch['user_dense']))

        # Concat all
        x = torch.cat(embs, dim=-1)
        assert x.shape[-1] == self._total_dim, f"dim mismatch: got {x.shape[-1]}, expected {self._total_dim}"

        logits = self.mlp(x).squeeze(-1)
        return logits


if __name__ == "__main__":
    # 单元测试
    import os
    import sys
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from paths import YEAR_DIR, ensure_dirs
    ensure_dirs()
    os.chdir(YEAR_DIR)
    from dataset import get_loaders

    train_loader, val_loader, vocab, info = get_loaders(batch_size=8)
    model = TAACBaseline(vocab, info, emb_dim=16, hidden_dims=(128, 64))

    n_params = sum(p.numel() for p in model.parameters())
    print(f"\n模型参数量: {n_params:,}")

    batch = next(iter(train_loader))
    logits = model(batch)
    print(f"logits shape: {logits.shape}")
    print(f"sample logits: {logits[:5].tolist()}")
    print(f"sample labels: {batch['label'][:5].tolist()}")
    print("✅ 模型前向传播 OK")
