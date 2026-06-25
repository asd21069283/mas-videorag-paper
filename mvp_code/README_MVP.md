# 多模态RAG视频系统 — MVP 阶段一：视频+文字 → 文字（音视频问答）

目标：在 AutoDL 单张 **4090(24G)** 上，用 **ms-swift** 对 **Qwen3-VL-4B-Instruct** 做
**LoRA（冻结视觉塔 ViT）** 微调，数据集 **MUSIC-AVQA**，把"看视频回答问题"这条链路先跑通。

> 一句话理解：给模型看一段乐器演奏的视频 + 问它"几样乐器在响?"，让它学会答"two"。
> 我们只训练"语言脑"(LLM 部分) 那几层小补丁(LoRA)，"眼睛"(ViT 视觉塔) 整个冻住不动 —— 这样 24G 显存才够、还便宜。

---

## 0. 关键坐标（已联网核对，2026-06）
| 项 | 值 | 来源 |
|---|---|---|
| 基座模型 (HF) | `Qwen/Qwen3-VL-4B-Instruct` | HuggingFace 模型卡 |
| 基座模型 (ModelScope，国内/AutoDL推荐) | `Qwen/Qwen3-VL-4B-Instruct` | 同名，走 ModelScope 下更快 |
| 训练框架 | `ms-swift >= 4.0` | swift 官方 Qwen3-VL Best Practice |
| transformers | `>= 4.57`（4.57 未正式发版时用 git 主分支） | HF 模型卡 / swift 文档 |
| 数据集 | MUSIC-AVQA (GeWu-Lab) | github.com/GeWu-Lab/MUSIC-AVQA |

官方文档（务必以这两个为准）：
- ms-swift Qwen3-VL 最佳实践: https://swift.readthedocs.io/en/latest/BestPractices/Qwen3-VL-Best-Practice.html
- MUSIC-AVQA 仓库: https://github.com/GeWu-Lab/MUSIC-AVQA

---

## 1. AutoDL 开实例
1. 卡型：先选 **RTX 4090 (24G)**，约 2 元/h。大实验再换 A100 80G。
2. 镜像：选 **PyTorch 2.x + CUDA 12.1（或 12.4）+ Python 3.10/3.11** 的官方基础镜像。
   - ⚠️ flash-attn 对 CUDA/torch 版本敏感，CUDA 12.1 兼容性最稳。
3. 系统盘默认即可，**数据盘挂到 `/root/autodl-tmp`**（MUSIC-AVQA 视频约 48G，必须放数据盘，别塞系统盘）。
4. 加速下载：AutoDL 内置学术加速 / 用 ModelScope 拉权重（国内不用翻墙）。

```bash
# AutoDL 开机后，开学术加速（拉 HF / github 时用，拉 ModelScope 不需要）
source /etc/network_turbo   # AutoDL 提供的命令，若镜像无此命令则忽略
```

---

## 2. 装环境
见 `requirements.txt`。核心命令：

```bash
# 用 ModelScope 源拉模型(国内快)，所以装 modelscope
pip install -U ms-swift modelscope
pip install "transformers>=4.57" "qwen_vl_utils>=0.0.14" "accelerate>=0.30"
pip install decord av                       # 视频解码后端
pip install -U "qwen_vl_utils[decord]"      # qwen 官方视频工具，decord 后端解帧快

# flash-attn 可选(能省显存/提速)，编译慢，4090够用可先跳过
# ⚠️ 必须匹配 torch/cuda；装不上就别装，去掉训练里的 --attn_impl flash_attn 即可
pip install flash-attn --no-build-isolation
```

> ⚠️ transformers 4.57 若尚未正式发版，按 HF 模型卡：
> `pip install git+https://github.com/huggingface/transformers`

---

## 3. 下载模型权重
```bash
# 方式A：ModelScope（AutoDL 国内推荐，不用学术加速）
export MODELSCOPE_CACHE=/root/autodl-tmp/ms_cache
modelscope download --model Qwen/Qwen3-VL-4B-Instruct --local_dir /root/autodl-tmp/Qwen3-VL-4B-Instruct

# 方式B：HuggingFace（需学术加速）
# export HF_HOME=/root/autodl-tmp/hf_cache
# huggingface-cli download Qwen/Qwen3-VL-4B-Instruct --local-dir /root/autodl-tmp/Qwen3-VL-4B-Instruct
```
> 训练脚本里直接写 `--model Qwen/Qwen3-VL-4B-Instruct` 也行，ms-swift 会自动从 ModelScope 拉（设 `USE_MODELSCOPE_HUB=1`）。下到本地目录是为了可控、不重复下载。

---

## 4. 下载 MUSIC-AVQA
- **视频**：官方放在 Google Drive / Baidu Drive(密码 cvpr)。Real(36.67GB) + Synthetic(11.59GB)，共 9288 个视频。
  - Google Drive: https://drive.google.com/drive/folders/1WAryZZE0srLIZG8VHl22uZ3tpbGHtsrQ
  - 全部解压汇总到一个目录，例如 `/root/autodl-tmp/MUSIC-AVQA/video/`（里面是 `00000238.mp4` 这种）。
  - ⚠️ Drive 大文件下 AutoDL 不一定顺，可在本地下好用 `scp`/AutoDL 网盘中转，或先用一个 mini 子集跑通流程。
- **标注 JSON**：仓库 `data/json/` 下 `avqa-train.json` / `avqa-val.json` / `avqa-test.json`。
  ```bash
  cd /root/autodl-tmp/MUSIC-AVQA
  git clone https://github.com/GeWu-Lab/MUSIC-AVQA.git repo
  # 标注就在 repo/data/json/*.json
  ```

标注一条的真实结构（已核对 avqa-train.json 实际内容）：
```json
{
  "video_id": "00000238",
  "question_id": 8,
  "type": "[\"Audio-Visual\", \"Counting\"]",
  "question_content": "How many instruments are sounding in the video?",
  "templ_values": "[]",
  "question_deleted": 0,
  "anser": "two"
}
```
> 注意几个坑：答案字段拼写是 **`anser`**（官方就这么拼，不是 answer）；`type`/`templ_values` 是**被转义的字符串**不是数组，要二次解析；有的问题含模板占位符，要用 `templ_values` 回填到 `question_content`（见 `prepare_musicavqa.py`）。

---

## 5. 数据预处理 → ms-swift JSONL
```bash
python prepare_musicavqa.py \
  --ann /root/autodl-tmp/MUSIC-AVQA/repo/data/json/avqa-train.json \
  --video_dir /root/autodl-tmp/MUSIC-AVQA/video \
  --out /root/autodl-tmp/MUSIC-AVQA/swift_train.jsonl

python prepare_musicavqa.py \
  --ann /root/autodl-tmp/MUSIC-AVQA/repo/data/json/avqa-val.json \
  --video_dir /root/autodl-tmp/MUSIC-AVQA/video \
  --out /root/autodl-tmp/MUSIC-AVQA/swift_val.jsonl
```

---

## 6. 启动训练
```bash
bash train_lora.sh
```
显存/时间预估（官方 Best Practice 实测，2*4B LoRA 约 2×21GiB / 12min 是小样例；MUSIC-AVQA 全量按规模放大）：
- **单 4090(24G)**：靠 `--freeze_vit true` + `FPS_MAX_FRAMES=8~16` + `VIDEO_MAX_PIXELS` 限制，单卡能跑。一个 epoch 视数据量从几小时起。
- 4090 跑不动（OOM）时按优先级降：
  1. `FPS_MAX_FRAMES=16 → 8 → 4`（少抽几帧，最直接省显存）
  2. `VIDEO_MAX_PIXELS=50176`（降每帧分辨率）
  3. `--max_length 4096 → 2048`
  4. 上 **QLoRA**：加 `--quant_method bnb --quant_bits 4`（4bit 量化基座，再省一截）
  5. `--gradient_checkpointing true`（用时间换显存，已默认开）
  6. 还不行 → 换 A100 80G。

花费预估：4090 ≈ 2 元/h。跑通流程(小子集)几块钱；全量一个 epoch 几十元量级。

---

## 7. 推理验证
```bash
python infer.py \
  --adapters /root/autodl-tmp/output/qwen3vl-musicavqa/vX-xxxx/checkpoint-xxx \
  --video /root/autodl-tmp/MUSIC-AVQA/video/00000238.mp4 \
  --question "How many instruments are sounding in the video?"
```

## 8. 合并 + 评测（可选）
```bash
python merge_and_eval.py \
  --adapters /root/autodl-tmp/output/qwen3vl-musicavqa/vX-xxxx/checkpoint-xxx \
  --val /root/autodl-tmp/MUSIC-AVQA/swift_val.jsonl
```

---

## 【已知坑 / 注意】

1. **transformers 版本**：Qwen3-VL 要 `>=4.57`。若 pip 装不到 4.57（未发版），按 HF 模型卡用 `pip install git+https://github.com/huggingface/transformers`。版本不对会直接报 `Unrecognized model / KeyError: 'qwen3_vl'`。
2. **答案字段拼写是 `anser`**（MUSIC-AVQA 官方就这么拼），不是 `answer`，写代码别想当然。`type` 和 `templ_values` 是**被转义的字符串**（如 `"[\"piano\"]"`），要再解析一次。
3. **问题模板回填**：部分问题含 `<Object>` 之类占位符，真实值在 `templ_values`，不回填的话模型看到的是占位符乱码。`prepare_musicavqa.py` 已处理。
4. **视频抽帧是显存命门**：`FPS_MAX_FRAMES` 默认 768、`VIDEO_MAX_PIXELS` 默认 602112，**不压根本跑不动 4090**。脚本已压到 16 帧 / 50176 像素。OOM 时优先继续降帧数。
5. **`--freeze_vit true` 后会有警告** `none of the inputs have requires_grad=True` —— 这是**正常**的（ViT 被冻了），别当报错。
6. **flash-attn 装不上很常见**（CUDA/torch/编译器不匹配）。装不上就**别加** `--attn_impl flash_attn`，4090 没它也能跑，只是慢一点。
7. **参数名以官方为准**：ms-swift 升级频繁，个别 flag 在不同大版本可能叫法不同（如老版本用 `--train_type` 而非 `--tuner_type`；`max_pixels` vs `VIDEO_MAX_PIXELS`）。⚠️ 若报 unrecognized arguments，去官方命令行参数全表核对：https://swift.readthedocs.io/en/latest/Instruction/Command-line-parameters.html
8. **MUSIC-AVQA 视频很大(~48G)且只在 Drive/百度盘**：AutoDL 直连 Google Drive 不稳。建议先用 `--require_video_exists` 拿**几百条子集跑通全链路**，确认无误再下全量。视频务必放数据盘 `/root/autodl-tmp`。
9. **评测的精确匹配偏严**：MUSIC-AVQA 答案是短词/数字，`merge_and_eval.py` 用规范化后精确匹配做近似准确率；正式汇报应按官方 9 类题型分别统计、并对同义答案做映射。
10. **省钱**：调试阶段随时 `swift infer` 用 CLI 交互验证，别反复全量训练；4090 约 2 元/h，跑通流程控制在个位数元。

---

## Sources
- [ms-swift Qwen3-VL Best Practice](https://swift.readthedocs.io/en/latest/BestPractices/Qwen3-VL-Best-Practice.html)
- [ms-swift Command-line parameters](https://github.com/modelscope/ms-swift/blob/main/docs/source_en/Instruction/Command-line-parameters.md)
- [Qwen/Qwen3-VL-4B-Instruct (HuggingFace)](https://huggingface.co/Qwen/Qwen3-VL-4B-Instruct)
- [GeWu-Lab/MUSIC-AVQA (GitHub)](https://github.com/GeWu-Lab/MUSIC-AVQA)
- [qwen-vl-utils (PyPI)](https://pypi.org/project/qwen-vl-utils/)
