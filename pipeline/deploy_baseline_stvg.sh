#!/bin/bash
# deploy_baseline_stvg.sh — 在 AutoDL 上部署 NeurIPS'25 Zero-Shot STVG (LLaVA_Next_STVG) 主对照
# 前提: 代码 tarball 已 scp 到 /root/autodl-tmp/LLaVA_Next_STVG.tar.gz (AutoDL 连不上 github)
# 用法: bash deploy_baseline_stvg.sh   (无卡模式即可做下载/装环境; 跑推理需 GPU)
set -e
export HF_ENDPOINT=https://hf-mirror.com
ROOT=/root/autodl-tmp
cd $ROOT

echo "=== 1) 解包代码 ==="
[ -d LLaVA_Next_STVG ] || tar xzf LLaVA_Next_STVG.tar.gz
cd LLaVA_Next_STVG

echo "=== 2) conda env (py3.11 + torch2.5.1 cu121) ==="
export PATH=/root/miniconda3/bin:$PATH
if ! conda env list | grep -q llava_stvg; then
  conda create -n llava_stvg python=3.11 -y
fi
source activate llava_stvg
pip install torch==2.5.1 torchvision==0.20.1 torchaudio==2.5.1 --index-url https://pypi.tuna.tsinghua.edu.cn/simple || \
pip install torch==2.5.1 torchvision==0.20.1 torchaudio==2.5.1
pip install -e ".[train]" -i https://pypi.tuna.tsinghua.edu.cn/simple
pip install "transformers==4.51.3" -i https://pypi.tuna.tsinghua.edu.cn/simple

echo "=== 3) 底模 LLaVA-NeXT-Video-7B-DPO (~15G, wget 走 hf-mirror; hf_hub 1.21 有bug别用) ==="
mkdir -p $ROOT/model_zoo/LLaVA-NeXT-Video-7B-DPO
python - <<'PY'
# 用本机预生成的文件清单; 若没有则现场列(公开仓 tree API 单页就够, 文件不多)
import urllib.request, json, os, subprocess
repo="lmms-lab/LLaVA-NeXT-Video-7B-DPO"
u=f"https://hf-mirror.com/api/models/{repo}/tree/main?recursive=true"
req=urllib.request.Request(u, headers={"User-Agent":"curl"})
files=[it["path"] for it in json.load(urllib.request.urlopen(req, timeout=60)) if it.get("type")=="file"]
print(len(files),"files")
dst="/root/autodl-tmp/model_zoo/LLaVA-NeXT-Video-7B-DPO"
for f in files:
    o=os.path.join(dst,f); os.makedirs(os.path.dirname(o), exist_ok=True)
    if os.path.exists(o) and os.path.getsize(o)>0: continue
    print("dl", f, flush=True)
    subprocess.run(["wget","-q","--tries=3","--timeout=120",
      f"https://hf-mirror.com/{repo}/resolve/main/{f}","-O",o], check=True)
print("model done")
PY

echo "=== 4) 它的数据(只下 annos + sam2候选框 + cache, 视频用我们已有的, 省~12G) ==="
mkdir -p stvg/data/hc-stvg2
python - <<'PY'
import urllib.request, json, os, subprocess, urllib.parse, re
repo="zaiquan/llava-stvg-data"
base=f"https://hf-mirror.com/api/datasets/{repo}/tree/main?recursive=true"
files=[]; cursor=None
while True:
    u=base+(f"&cursor={urllib.parse.quote(cursor)}" if cursor else "")
    req=urllib.request.Request(u, headers={"User-Agent":"curl"})
    r=urllib.request.urlopen(req, timeout=60)
    for it in json.load(r):
        if it.get("type")=="file": files.append(it["path"])
    link=r.headers.get("Link",""); m=re.search(r'cursor=([^&>]+)', link)
    if 'rel="next"' in link and m: cursor=urllib.parse.unquote(m.group(1))
    else: break
want=[f for f in files if f.startswith("hc-stvg2/") and not f.startswith("hc-stvg2/v2_video/")]
print("要下", len(want), "个(跳过v2_video)")
for i,f in enumerate(want):
    o=os.path.join("/root/autodl-tmp/LLaVA_Next_STVG/stvg/data", f)
    os.makedirs(os.path.dirname(o), exist_ok=True)
    if os.path.exists(o) and os.path.getsize(o)>0: continue
    subprocess.run(["wget","-q","--tries=3","--timeout=120",
      f"https://hf-mirror.com/datasets/{repo}/resolve/main/{f}","-O",o], check=True)
    if (i+1)%500==0: print(i+1, flush=True)
print("data done")
PY

echo "=== 5) 视频: 按 stem 软链我们已有的 1893 个到它期望的位置 ==="
mkdir -p stvg/data/hc-stvg2/v2_video
python - <<'PY'
import os, glob
src="/root/autodl-tmp/hcstvg2/v2_video"
dst="/root/autodl-tmp/LLaVA_Next_STVG/stvg/data/hc-stvg2/v2_video"
n=0
for p in glob.glob(f"{src}/*"):
    t=os.path.join(dst, os.path.basename(p))
    if not os.path.exists(t): os.symlink(p, t); n+=1
print("symlinked", n)
PY

echo "=== 6) 改跑批脚本路径(模型指到 /root/autodl-tmp/model_zoo) ==="
sed -i 's#/home/zaiquyang2/scratch/mllm/model_zoo/LLaVA-NeXT-Video-7B-DPO#/root/autodl-tmp/model_zoo/LLaVA-NeXT-Video-7B-DPO#' stvg/run_hcstvg2.sh || true
df -h /root/autodl-tmp | tail -1
echo "=== DEPLOY DONE. GPU 模式下: cd LLaVA_Next_STVG && bash stvg/run_hcstvg2.sh ==="
