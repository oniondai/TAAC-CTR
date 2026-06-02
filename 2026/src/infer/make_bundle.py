"""
打包 infer/ 目录为 zip + 生成 manifest

按 BestDIN submit_eval.mjs 的协议生成 manifest.json
平台吃单独文件上传, 但 zip 方便你直接拖到网页
"""
import os
import json
import hashlib
import zipfile
import datetime
import sys

# 仓库内统一路径
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from paths import BUNDLE_DIR, MODELS_DIR, YEAR_DIR, ensure_dirs

# 源: 把训练好的模型/代码 (从 models/) 打包到 bundle/ 给平台
INFER_DIR = BUNDLE_DIR
OUTPUT_DIR = BUNDLE_DIR


def sha256_file(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(8192), b""):
            h.update(chunk)
    return h.hexdigest()


def build_manifest():
    """按 mjs 协议生成 manifest"""
    files_info = []
    for fname in sorted(os.listdir(INFER_DIR)):
        fpath = os.path.join(INFER_DIR, fname)
        if not os.path.isfile(fpath):
            continue
        size = os.path.getsize(fpath)
        sha = sha256_file(fpath)
        files_info.append({
            "name": fname,
            "local_path": fpath,
            "size": size,
            "sha256": sha,
            "content_type": (
                "application/json" if fname.endswith(".json") else
                "application/zip" if fname.endswith(".zip") else
                "text/x-python" if fname.endswith(".py") else
                "application/octet-stream" if fname.endswith((".pth", ".pt")) else
                "application/octet-stream"
            )
        })
    return {
        "mould_name": "our_v2_baseline",
        "created_at": datetime.datetime.now().isoformat(),
        "purpose": "Eval inference bundle for TAAC2026",
        "expected_files": ["dataset.py", "model.py", "model_v2.py", "infer.py",
                           "best_model.pth", "vocab.json", "info.json"],
        "files": files_info,
        "total_files": len(files_info),
        "total_size": sum(f["size"] for f in files_info),
        "upload_protocol": "Each file uploaded separately to COS as local--<uuid>/<filename>",
    }


def make_zip():
    """把 infer/ 打包为 zip, 直接拖到 algo.qq.com"""
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    zip_path = os.path.join(OUTPUT_DIR, f"taac2026_infer_v2_{timestamp}.zip")
    with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as zf:
        for fname in sorted(os.listdir(INFER_DIR)):
            fpath = os.path.join(INFER_DIR, fname)
            if os.path.isfile(fpath):
                # 平台要求是顶层文件, 不用 infer/ 前缀
                zf.write(fpath, arcname=fname)
                print(f"  + {fname} ({os.path.getsize(fpath):,} bytes)")
    return zip_path


def main():
    print("📦 打包 infer/ 目录")
    print(f"   源: {INFER_DIR}")
    print(f"   输出: {OUTPUT_DIR}\n")

    # 1. 生成 manifest
    print("📋 生成 manifest.json (按 submit_eval.mjs 协议)")
    manifest = build_manifest()
    manifest_path = os.path.join(OUTPUT_DIR, "infer_manifest.json")
    with open(manifest_path, "w") as f:
        json.dump(manifest, f, indent=2)
    print(f"   ✓ {manifest_path}")
    print(f"   ✓ 共 {manifest['total_files']} 个文件, "
          f"总大小 {manifest['total_size']:,} bytes "
          f"({manifest['total_size']/1024/1024:.2f} MB)\n")

    # 2. 打包 zip
    print("🗜️  打包 zip (方便直接上传到 algo.qq.com 网页)")
    zip_path = make_zip()
    zip_size = os.path.getsize(zip_path)
    print(f"\n   ✓ {zip_path}")
    print(f"   ✓ 大小: {zip_size:,} bytes ({zip_size/1024/1024:.2f} MB)")

    # 3. 校验
    print("\n🔍 校验 zip 内容...")
    with zipfile.ZipFile(zip_path) as zf:
        zf.testzip()
        names = zf.namelist()
    print(f"   ✓ zip 完整, 包含 {len(names)} 个文件:")
    for n in sorted(names):
        info = zf.getinfo(n)
        print(f"      - {n} ({info.file_size:,} bytes)")

    # 4. 输出最终的 README (上平台时给评审批注)
    print("\n📝 生成上传说明 README")
    readme_content = f"""# TAAC2026 提交 Bundle (v2)

## 包含的文件

```
{chr(10).join(f"  - {f['name']} ({f['size']:,} bytes)" for f in manifest['files'])}
```

## 上传方式 (二选一)

### 方式 1: 网页直接拖 (推荐)

1. 打开 https://algo.qq.com 进入"我的提交"页
2. 上传这个 zip: `{os.path.basename(zip_path)}`
3. 选 MouldId (你已注册的模型 ID)
4. 提交

### 方式 2: PowerShell 上传 (Windows)

按 TAIJI.md 用 `submit_eval.ps1`:

```powershell
powershell -ExecutionPolicy Bypass -File Automation/ps1/submit_eval.ps1 \\
  -MouldId <你的模型ID> \\
  -Name "our_v2_baseline" \\
  -InferDir "{INFER_DIR}" \\
  -CookieFile "taiji-output/secrets/taiji-cookie.txt" \\
  -Execute
```

## 注意事项

1. **必须用 v2 模型** (`best_model.pth` 是 v2 权重)
2. **不要改 infer 目录的文件名** (平台按名查找)
3. **如果平台报错 ImportError**, 需要加 `requirements.txt` 到 zip 里

## 本地测试成绩

- Val AUC (demo_1000): **0.6840**
- 本地模拟 Eval AUC: **0.9671** (注: 训练/测试同源, 数据泄漏)
- 真实平台 AUC 估算: 0.5x-0.6x (受数据规模和分布影响)
"""
    readme_path = os.path.join(OUTPUT_DIR, "UPLOAD_README.md")
    with open(readme_path, "w") as f:
        f.write(readme_content)
    print(f"   ✓ {readme_path}")

    print(f"\n{'='*70}")
    print(f"🎉 打包完成！")
    print(f"{'='*70}")
    print(f"   zip:        {zip_path}")
    print(f"   manifest:   {manifest_path}")
    print(f"   readme:     {readme_path}")


if __name__ == "__main__":
    main()
