# 🚀 TAAC-CTR

腾讯广告算法大赛（TAAC 2026 × KDD Cup 2026）CTR 预估项目。

基于 **DIN (Deep Interest Network)** 改进的工业级 Baseline，端到端跑通了
**数据下载 → 模型训练 → 本地推理 → 平台上传包** 全流程。

> 训练平台: [taiji.algo.qq.com](https://algo.qq.com) (TAIJI)
> 本仓库: 在本地 CPU 上跑通，1000 行 demo 数据，~3 分钟训练 30 epoch

---

## ✨ 核心特性

- **🎯 DIN + 3 个涨分改造 (v2 模型)**
  - Hash Embedding (multi-bucket sum) — 解决 user/item ID 高基数
  - Sample Time Encoder — Hour + Weekday + IsWeekend
  - Pair Encoding — FID 62~66 user_int/dense 共享编码
- **🛡️ OOM-Safe 数据加载**: 内置 `safe_idx` 强力护盾，拦截脏数据
- **⚡ 纯相对路径**: 所有脚本从仓库根自动推导，clone 到任意机器都能跑
- **📦 平台一键打包**: `make_bundle.py` 生成 zip + manifest，腾讯 TAIJI 直接吃

---

## 📂 项目结构

```
TAAC-CTR/
├── README.md                      # 本文件
├── .gitignore
└── 2026/
    ├── requirements.txt           # Python 依赖
    ├── scripts/
    │   └── run.sh                 # ⭐ 一键全流程入口
    └── src/
        ├── paths.py               # 统一路径工具 (所有脚本的根)
        ├── data/
        │   ├── download_demo.py   # 从 HF 下载 TAAC2026/data_sample_1000
        │   └── gen_mock_data.py   # 生成 Tenrec 格式 mock CSV
        ├── model/
        │   ├── dataset.py         # PyTorch Dataset + vocab 构建
        │   ├── model.py           # v1 模型 (baseline DIN)
        │   ├── model_v2.py        # v2 模型 (Hash + Time + Pair 改进)
        │   ├── main.py            # v1 训练脚本
        │   └── main_v2.py         # v2 训练脚本 (推荐)
        └── infer/
            ├── infer.py           # 推理 (读 ckpt → 写 predictions.json)
            └── make_bundle.py     # 打包 zip + manifest 给平台上传
```

**不**入库的产物 (`.gitignore` 已配)：
- `2026/data_official.parquet` (38 MB demo 数据)
- `2026/models/*.pth` (训练好的权重)
- `2026/bundle/` (平台上传包 + predictions.json)
- `2026/vocab.json` / `info.json` (训练时自动生成)

---

## ⚙️ 环境依赖

- Python 3.8+
- PyTorch 1.12+ (CPU 即可)
- 详见 `2026/requirements.txt`

```bash
# 推荐: 用清华源装, 速度快
pip install -r 2026/requirements.txt -i https://pypi.tuna.tsinghua.edu.cn/simple

# 如果在中国大陆, 下载 HF 数据需要先设代理
export HTTP_PROXY=http://127.0.0.1:8118
export HTTPS_PROXY=http://127.0.0.1:8118
```

---

## 🚀 快速开始

### 方式 1: 一键全流程 (推荐)

```bash
cd TAAC-CTR/2026
bash scripts/run.sh all
```

这会按顺序执行:
1. 从 HuggingFace 下载 demo 数据 (1000 行, ~40 MB)
2. 训练 v2 模型 (CPU ~3 分钟, 30 epoch)
3. 拿训练集前 20% 当 test 跑推理
4. 打包平台上传 zip + manifest

### 方式 2: 分步执行

```bash
cd TAAC-CTR/2026

# 1. 只下载数据
bash scripts/run.sh data

# 2. 训练 (v2 推荐, 也可换 v1)
bash scripts/run.sh train v2
# bash scripts/run.sh train v1

# 3. 推理 (需要先准备测试数据, 默认用训练集 20% 模拟)
bash scripts/run.sh infer

# 4. 打包平台上传 bundle
bash scripts/run.sh bundle
```

### 方式 3: 直接调 Python (调试时)

```bash
cd TAAC-CTR/2026

python src/data/download_demo.py
python src/model/main_v2.py
python src/model/model_v2.py    # 跑前向单测

# 推理需要三个环境变量 (跟 TAIJI 平台约定一致)
MODEL_OUTPUT_PATH=2026/models \
EVAL_DATA_PATH=/path/to/test_parquet_dir \
EVAL_RESULT_PATH=2026/bundle \
  python src/infer/infer.py
```

---

## 🧠 模型架构

### v1 (baseline) - `model.py`
普通 DIN: Embedding → Flatten → MLP → Sigmoid

### v2 (推荐) - `model_v2.py`
在 v1 基础上加 3 个涨分点:

| 改进 | 描述 | 涨点预期 |
|---|---|---|
| **Hash Embedding** | 多桶 hash 替代高基数 ID 的 Embedding | +1~2% |
| **Sample Time Encoder** | 绝对时间上下文 (Hour + Weekday + IsWeekend) | +0.5~1% |
| **Pair Encoding** | FID 62~66 的 user_int 和 user_dense 共享编码 (element-wise product) | +0.5~1% |

参数量: ~3.6M (CPU 训练无压力)

---

## 📊 数据格式

**输入**: HuggingFace `TAAC2026/data_sample_1000` parquet 文件
- 1000 行, 120 列
- 字段类型: `user_id`, `item_id`, `label_type` (1=负/2=正), `timestamp`, `user_int_feats_*` (46个), `item_int_feats_*` (14个), `user_dense_feats_*` (10个), `*_seq_*` (45个)

**输出**: `bundle/predictions.json`
```json
{
  "predictions": {
    "12635809": 0.5094,
    "9634495": 0.5103,
    ...
  }
}
```

---

## 📦 平台上传 (TAIJI)

`make_bundle.py` 会:
1. 把 `2026/models/best_model_v2.pth` + `vocab.json` + `info.json` 复制到 `2026/bundle/`
2. 同时把推理脚本 (`dataset.py` / `model.py` / `model_v2.py` / `infer.py`) 拷过去
3. 打成 `taac2026_infer_v2_<时间戳>.zip`
4. 生成 `infer_manifest.json` (每个文件的 sha256)
5. 生成 `UPLOAD_README.md` (给评审批注用)

然后在 [taiji.algo.qq.com](https://algo.qq.com) 的"我的提交"页:
- 上传 zip
- 选 MouldId (你注册的模型 ID)
- 提交评测

---

## 🛠️ 常见问题

**Q: 训练要多久？**
A: 1000 行 demo 数据, CPU 30 epoch 大约 3 分钟。全量数据 (百万级) 视平台机器配置 30 分钟~2 小时。

**Q: `from paths import` 报错 ModuleNotFoundError？**
A: 不要直接 `cd src/model && python main_v2.py`，要在仓库根的 `2026/` 下执行。所有脚本都用了 `sys.path.insert(0, ...)` 自动找 paths.py。

**Q: 我在中国大陆，下载 HF 数据失败？**
A: 先 `export HTTP_PROXY=http://127.0.0.1:8118 && export HTTPS_PROXY=...`，或者用 v2ray/ss 代理。

**Q: 怎么换成全量训练数据？**
A: 平台训练时数据会通过环境变量 `TRAIN_DATA_PATH` 注入，把 `dataset.py` 里的 `LOCAL_PARQUET` 改成读 `TRAIN_DATA_PATH/*.parquet` 即可。

---

## 📈 已知成绩

- Val AUC (demo_1000, 3 epoch 冒烟): 0.5384
- Val AUC (demo_1000, 30 epoch 完整训练): 待补
- 真实平台 AUC: 取决于数据规模与分布

---

## 📝 许可证

仅供学习交流。
