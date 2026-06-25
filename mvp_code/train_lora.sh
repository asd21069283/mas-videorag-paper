#!/usr/bin/env bash
# train_lora.sh
# ms-swift LoRA(冻结ViT) 微调 Qwen3-VL-4B-Instruct on MUSIC-AVQA, 目标单张 4090(24G) 可跑。
# 参数依据官方 Qwen3-VL Best Practice:
#   https://swift.readthedocs.io/en/latest/BestPractices/Qwen3-VL-Best-Practice.html
set -e

export USE_MODELSCOPE_HUB=1                       # 用 ModelScope 拉权重(AutoDL/国内免翻墙)
export PYTORCH_CUDA_ALLOC_CONF='expandable_segments:True'  # 缓解显存碎片, 防 OOM

# ===== 省显存的视频抽帧/分辨率控制(env 变量, ms-swift 读取) =====
# 这几个是 4090 能不能跑起来的命门:
export FPS=2.0                 # 每秒抽 2 帧(默认2.0)。视频越长帧越多越吃显存
export FPS_MIN_FRAMES=4        # 单视频最少抽 4 帧
export FPS_MAX_FRAMES=16       # ⚠️ 单视频最多抽 16 帧(默认768, 必须压!) OOM 就降到 8/4
export VIDEO_MAX_PIXELS=50176  # 每帧最大像素(默认602112), 压低省显存; 还紧就 ->50176/32256
export VIDEO_MAX_TOKEN_NUM=128 # 单个视频最多占的视觉 token 数, 限上限防爆

CUDA_VISIBLE_DEVICES=0 \
swift sft \
    --model Qwen/Qwen3-VL-4B-Instruct \
    --dataset /root/autodl-tmp/MUSIC-AVQA/swift_train.jsonl \
    --val_dataset /root/autodl-tmp/MUSIC-AVQA/swift_val.jsonl \
    --tuner_type lora \
    --lora_rank 8 \
    --lora_alpha 32 \
    --target_modules all-linear \
    --freeze_vit true \
    --freeze_aligner true \
    --torch_dtype bfloat16 \
    --max_length 4096 \
    --num_train_epochs 1 \
    --per_device_train_batch_size 1 \
    --gradient_accumulation_steps 16 \
    --learning_rate 1e-4 \
    --gradient_checkpointing true \
    --save_steps 200 \
    --eval_steps 200 \
    --logging_steps 5 \
    --warmup_ratio 0.05 \
    --dataloader_num_workers 4 \
    --output_dir /root/autodl-tmp/output/qwen3vl-musicavqa

# ---- 关键参数为什么这么设 ----
#  --tuner_type lora            : 只训 LoRA 小补丁, 不动全量权重(显存/磁盘都省)
#  --lora_rank 8 / alpha 32     : 官方推荐起步组合, 任务难再调大 rank
#  --target_modules all-linear  : 对所有线性层挂 LoRA(官方推荐写法)
#  --freeze_vit true            : 【冻结视觉塔】—— 任务要求, 也是 24G 跑得动的关键
#  --freeze_aligner true        : 视觉->语言的对齐层也冻, 进一步省显存
#  --per_device_train_batch_size 1 + grad_accum 16 : 真实 batch=16, 但显存只按 1 个算
#  --bf16(torch_dtype bfloat16) : 4090 支持, 比 fp16 数值稳
#  --gradient_checkpointing true: 用计算换显存(反向时重算激活), VLM 几乎必开
#
# ---- 还 OOM? 取消注释开 QLoRA(4bit 量化基座, 再省一大截) ----
#   --quant_method bnb \
#   --quant_bits 4 \
#   (需先 pip install bitsandbytes)
#
# ---- 装了 flash-attn 才加(没装就别加, 否则报错) ----
#   --attn_impl flash_attn
#
# ⚠️ 以上所有 flag/env 名以官方最新文档为准, 版本升级可能改名:
#    https://swift.readthedocs.io/en/latest/BestPractices/Qwen3-VL-Best-Practice.html
#    命令行参数全表: https://swift.readthedocs.io/en/latest/Instruction/Command-line-parameters.html
