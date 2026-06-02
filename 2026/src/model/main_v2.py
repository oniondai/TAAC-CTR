"""
TAAC2026 v2 训练脚本
- 30 epochs, 跟 v1 一样的 batch/early stop
- 模型换成 TAACv2
- 验证 AUC, 保存 best_model_v2.pth
"""
import os
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import json
from sklearn.metrics import roc_auc_score, accuracy_score

os.environ['OMP_NUM_THREADS'] = '4'

# 把 src/ 加到 path, 这样 from paths/... 能找到 (paths.py 在 src/)
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from dataset import get_loaders
from model_v2 import TAACv2
from paths import CKPT_PATH_V2, INFO_PATH


def train_and_eval():
    BATCH_SIZE = 64
    EPOCHS = 30
    LR = 0.005
    EMB_DIM = 16
    PATIENCE = 8
    SEED = 42

    torch.manual_seed(SEED)
    np.random.seed(SEED)

    print("🚀 准备启动 TAAC2026 v2 训练...")
    print(f"   Batch size: {BATCH_SIZE}, Epochs: {EPOCHS}, LR: {LR}\n")

    train_loader, val_loader, vocab, info = get_loaders(batch_size=BATCH_SIZE)
    print(f"\n📊 数据: {len(train_loader)} train batches, {len(val_loader)} val batches")
    print(f"   特征: {len(info['user_int_keys'])} user_int + {len(info['item_int_keys'])} item_int + "
          f"{len(info['seq_keys'])} seq + dense={info['dense_dim']}\n")

    DEVICE = torch.device("cpu")
    model = TAACv2(vocab, info, emb_dim=EMB_DIM, hidden_dims=(128, 64), dropout=0.3).to(DEVICE)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"🤖 v2 模型: {n_params:,} 参数, 设备: {DEVICE}\n")

    pos_weight = torch.tensor([(876 / 124)], dtype=torch.float32)
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    optimizer = optim.Adam(model.parameters(), lr=LR, weight_decay=1e-5)

    best_auc = 0.0
    patience_counter = 0
    best_state = None

    for epoch in range(EPOCHS):
        model.train()
        total_loss = 0.0
        n_batches = 0

        for batch_idx, batch in enumerate(train_loader):
            labels = batch['label'].to(DEVICE)
            optimizer.zero_grad()
            logits = model(batch)
            loss = criterion(logits, labels)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            n_batches += 1

        avg_loss = total_loss / n_batches

        # Val
        model.eval()
        all_labels = []
        all_preds = []
        with torch.no_grad():
            for batch in val_loader:
                labels = batch['label'].to(DEVICE)
                logits = model(batch)
                probs = torch.sigmoid(logits).numpy()
                all_labels.extend(labels.numpy())
                all_preds.extend(probs)

        all_labels = np.array(all_labels)
        all_preds = np.array(all_preds)

        try:
            val_auc = roc_auc_score(all_labels, all_preds)
            preds_bin = (all_preds > 0.5).astype(int)
            val_acc = accuracy_score(all_labels, preds_bin)
        except ValueError:
            val_auc = 0.5
            val_acc = 0.0

        improved = ""
        if val_auc > best_auc:
            best_auc = val_auc
            patience_counter = 0
            torch.save(model.state_dict(), CKPT_PATH_V2)
            improved = " ✨ best, saved"
        else:
            patience_counter += 1

        print(f"🏆 Epoch {epoch+1:02d}/{EPOCHS} | Train Loss: {avg_loss:.4f} | "
              f"Val AUC: {val_auc:.4f} | Val Acc: {val_acc:.4f}{improved}", flush=True)

        if patience_counter >= PATIENCE:
            print(f"\n🛑 触发早停, 最优 Val AUC: {best_auc:.4f}")
            break

    print(f"\n{'='*60}")
    print(f"🎯 v2 训练结束! 最优 Val AUC: {best_auc:.4f}")
    print(f"{'='*60}")

    # 保存 info.json 以便 infer 复用
    info_serializable = {
        'user_int_keys': info['user_int_keys'],
        'item_int_keys': info['item_int_keys'],
        'user_dense_keys': info['user_dense_keys'],
        'seq_keys': info['seq_keys'],
        'dense_dim': info['dense_dim'],
    }
    with open(INFO_PATH, "w") as f:
        json.dump(info_serializable, f, indent=2)
    print(f"✅ info.json 已保存")


if __name__ == "__main__":
    import sys
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from paths import YEAR_DIR, ensure_dirs
    ensure_dirs()
    os.chdir(YEAR_DIR)
    train_and_eval()
