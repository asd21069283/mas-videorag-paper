#!/bin/bash
# 下次开机一键装环境(在 AutoDL 上跑)。无卡模式即可(纯下载/装包, 不用GPU)。
set -e
export PATH=/root/miniconda3/bin:$PATH
export HF_HOME=/root/autodl-tmp/hf
export HF_ENDPOINT=https://hf-mirror.com
mkdir -p /root/autodl-tmp/hf

echo "=== 核心: diffusers 主分支(FLUX-Kontext需要) + transformers 等 ==="
pip install -q -U "git+https://github.com/huggingface/diffusers.git"
pip install -q -U transformers accelerate safetensors sentencepiece protobuf pillow
pip install -q -U qwen-vl-utils decord av
echo "=== 可选: VQAScore(较大, 不急可跳过) ==="
pip install -q t2v-metrics || echo "t2v-metrics 装失败可跳过, 先用CLIP指标"

echo "=== Grounding-DINO 权重(transformers内置类, 预拉权重) ==="
python - <<'PY'
import os
os.environ.setdefault("HF_ENDPOINT","https://hf-mirror.com")
from huggingface_hub import snapshot_download
try:
    snapshot_download("IDEA-Research/grounding-dino-base", allow_patterns=["*.json","*.safetensors","*.txt","*.bin"])
    print("grounding-dino-base ok")
except Exception as e:
    print("预拉失败(运行时会自动下):", str(e)[:120])
PY

echo "=== 验证关键导入 ==="
python -c "import diffusers,transformers,decord,qwen_vl_utils;from diffusers import FluxKontextPipeline;from transformers import AutoModelForZeroShotObjectDetection;print('imports OK; diffusers',diffusers.__version__)"
echo SETUP_NEXT_DONE
