# 评测脚本（eval/）

对应 issue #6（评测体系）+ #11（VKIG-Bench）。VKIG 答案 = 图文交错 + 每张图带 `(keyframe_ts, bbox, object)` 溯源。

## 四个维度

| 维度 | 脚本 | 需要 | 本地状态 |
|---|---|---|---|
| **时空定位**（图是否对应正确时刻/物体） | `spatiotemporal_iou.py` | 纯 Python | ✅ 已本地自测通过 |
| **图↔文一致**（图是否符合指令/文本） | `image_metrics.py vqascore` | GPU + `t2v-metrics` | 脚本就绪，待 GPU 跑 |
| **关键帧一致**（图↔溯源关键帧主体，核心维度） | `image_metrics.py dreamsim` | GPU + `dreamsim` | 脚本就绪，待 GPU 跑 |
| **画质** | `image_metrics.py fid` | GPU + `pytorch-fid` | 脚本就绪，待 GPU 跑 |
| 文本正确 / 总体 | （用 GPT-4o-judge，另接） | API | 待接 |

## 快速自测（无需 GPU）
```bash
python eval/spatiotemporal_iou.py --selftest
python mvp_code/vstar_to_vkig.py --selftest
```

## 正式评测
```bash
# 1) 时空定位
python eval/spatiotemporal_iou.py --pred pred.jsonl --gold vkig.jsonl --siou_thr 0.5
# 2) 图文一致 / 主体一致 / 画质 (GPU)
pip install t2v-metrics dreamsim pytorch-fid
python eval/image_metrics.py vqascore --pairs pairs.jsonl      # {"image","text"}
python eval/image_metrics.py dreamsim --pairs pairs.jsonl      # {"image","ref"}
python eval/image_metrics.py fid --gen_dir gen/ --ref_dir ref/
```

## 数据准备
- `mvp_code/vstar_to_vkig.py`：V-STaR → VKIG schema（关键帧取时间区间中点最近的框）。
- gold/pred 均为 VKIG jsonl；`spatiotemporal_iou.py` 按 `video` 字段对齐。

> 指标取向：DreamSim/时空定位 是相对非视频基准的**独有维度**（护城河）；VQAScore 比 CLIPScore 更可靠；FID 需足够样本量才稳。
