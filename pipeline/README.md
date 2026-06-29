# pipeline/ — 下次开机即用：完整 Video+Text → Image+Text 流水线

把"合成单图玩具"升级成"真实视频原型"。一段真实视频 + 一个问题，跑：
**① Q-driven 选帧 → ② Grounding-DINO 定位 → ③ Qwen3-VL 理解 → ④ FLUX-Kontext 基于关键帧生成 → ⑤ 评测**

## 下次开机的确切步骤

```bash
# 0) 无卡模式即可做 1-2; 跑 3(GPU) 时切 GPU 开机
ssh -i ~/.ssh/id_ed25519 -p <端口> root@<host>          # (我来连)
cd /root/autodl-tmp/mas-videorag-paper && git pull       # 拉最新代码

# 1) 装环境(无卡模式, 0.01元/h)
bash pipeline/setup_next.sh

# 2) 下 V-STaR 子集(无卡模式; 先下 20 条试)
python pipeline/download_vstar_subset.py --n 20 --out /root/autodl-tmp/vstar

# 3) 切 GPU 开机, 跑一个真实样本端到端
export PATH=/root/miniconda3/bin:$PATH
python pipeline/vkig_pipeline.py \
  --video /root/autodl-tmp/vstar/<某视频> \
  --query "<该样本的问题>" \
  --object "<问题里的目标物体, 如 red car>" \
  --out_dir /root/autodl-tmp/run1
# 产物: run1/keyframe.png, generated.png, result.json
```

## 各步骤说明
| 步 | 函数 | 模型 | 显存 |
|---|---|---|---|
| ① 选帧 | `select_keyframes` (AKS式: 相关性+时序覆盖) | CLIP-B/32 | 低 |
| ② 定位 | `ground_object` | Grounding-DINO-base | ~2-4G |
| ③ 理解 | `understand` | Qwen3-VL-4B | ~10G |
| ④ 生成 | `generate_image` (FLUX优先, SDXL回退) | FLUX.1-Kontext / SDXL-Turbo | FLUX需开 cpu_offload |
| ⑤ 评测 | `clip_eval` + `eval/spatiotemporal_iou` | CLIP | 低 |

> 各模型**按需加载并释放**(24G 装不下全部同时在显存), 单样本顺序跑没问题。

## 无需 GPU 的自测(本机已验证)
```bash
python pipeline/vkig_pipeline.py --selftest     # 选帧覆盖逻辑
```

## 已知/版本敏感点(首跑留意, 代码里已标 ⚠️)
- FLUX-Kontext 需 **diffusers 主分支**(setup 已装 git 版)；24G 必须 `enable_model_cpu_offload()`(已内置)。
- Grounding-DINO 的 `post_process_grounded_object_detection` 在不同 transformers 版本参数名可能是 `threshold` vs `box_threshold`，报错就改名。
- V-STaR 文件布局以仓库实际为准；`download_vstar_subset.py` 自动探测，若没下到视频会打印提示。
- DreamSim 权重被墙挡，暂用 CLIP 等价指标；VQAScore(t2v-metrics)较大，可后接。

## 跑通一条后的下一步(向论文推进)
1. 批量跑 50 条 → 形成 VKIG dev 集 + 出我们方法的指标。
2. 跑一个 baseline(Open-o3-Video / VideoRAG) → 出对比数(论文命根子)。
3. 上 VQAScore + 解决 DreamSim → 正式指标。
4. 接入老师定的 Q1 场景的自建数据。
