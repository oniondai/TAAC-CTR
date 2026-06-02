"""
TAAC2026 官方 demo Baseline 训练主循环
- 1000 样本，CPU 训练
- 二分类: label_type (1->0, 2->1) → BCE loss
- 早停 + best model 保存
"""
import os
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from sklearn.metrics import roc_auc_score, accuracy_score

# 强制单线程，CPU 训练更稳
os.environ['OMP_NUM_THREADS'] = '4'

# 把 src/ 加到 path, 这样 from paths/... 能找到 (paths.py 在 src/)
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from dataset import get_loaders
from model import TAACBaseline
from paths import CKPT_PATH_V1


def train_and_eval():
    BATCH_SIZE = 64
    EPOCHS = 30
    LR = 0.005
    EMB_DIM = 16
    PATIENCE = 8
    SEED = 42

    torch.manual_seed(SEED)
    np.random.seed(SEED)

    print("🚀 准备启动 TAAC2026 官方 demo 训练...")
    print(f"   Batch size: {BATCH_SIZE}, Epochs: {EPOCHS}, LR: {LR}")
    print()

    # 加载数据
    train_loader, val_loader, vocab, info = get_loaders(batch_size=BATCH_SIZE)
    print(f"\n📊 数据加载完成:")
    print(f"   训练集 batch: {len(train_loader)}")
    print(f"   验证集 batch: {len(val_loader)}")
    print(f"   特征维度: {len(info['user_int_keys'])} user_int + {len(info['item_int_keys'])} item_int + "
          f"{len(info['seq_keys'])} seq + dense={info['dense_dim']}")
    print()

    # 模型
    DEVICE = torch.device("cpu")
    model = TAACBaseline(vocab, info, emb_dim=EMB_DIM, hidden_dims=(128, 64), dropout=0.3).to(DEVICE)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"🤖 模型: {n_params:,} 参数, 设备: {DEVICE}\n")

    # 类别不均衡：label=1 占比 12.4%，用 pos_weight
    pos_weight = torch.tensor([(876 / 124)], dtype=torch.float32)
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    optimizer = optim.Adam(model.parameters(), lr=LR, weight_decay=1e-5)

    best_auc = 0.0
    patience_counter = 0

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

        # 验证
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

        print(f"🏆 Epoch {epoch+1:02d}/{EPOCHS} | Train Loss: {avg_loss:.4f} | "
              f"Val AUC: {val_auc:.4f} | Val Acc: {val_acc:.4f}")

        if val_auc > best_auc:
            best_auc = val_auc
            patience_counter = 0
            torch.save(model.state_dict(), CKPT_PATH_V1)
            print(f"   ✨ 发现更高 AUC, 模型已保存")
        else:
            patience_counter += 1
            print(f"   ⚠️ AUC 未提升, 耐心值: {patience_counter}/{PATIENCE}")

        if patience_counter >= PATIENCE:
            print(f"\n🛑 触发早停, 最优 Val AUC: {best_auc:.4f}")
            break

    print(f"\n{'='*60}")
    print(f"🎯 训练结束! 最优 Val AUC: {best_auc:.4f}")
    print(f"{'='*60}")


if __name__ == "__main__":
    import sys
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from paths import YEAR_DIR, CKPT_PATH_V1, ensure_dirs
    ensure_dirs()
    os.chdir(YEAR_DIR)
    train_and_eval()
