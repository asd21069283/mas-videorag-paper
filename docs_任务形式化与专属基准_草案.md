# 任务形式化 & 专属基准设计（草案 v0.1，2026-06-26）

> 目的：把"视频+文字→图+文字"**形式化**，并设计一个**带视频输入 + 关键帧一致性**评测的专属小基准——这是相对统一大模型（BAGEL/Show-o2）的护城河（见 `docs_更新_2026-06-26.md` §一风险）。
> 状态：草案。**场景锚定待老师定（问题清单第二轮 Q1）**，本草案先按推荐场景【影视/短剧】写，确认后再细化。对应 issue #11。

---

## 1. 任务形式化（Task Definition）

**输入**：视频片段 `V` + 文字指令/问题 `Q`。
**输出**：图文交错答案 `A = [t_1, img_1, t_2, img_2, …]`，其中
- `t_i` 为文本片段；
- `img_i` 为**基于 `V` 的某个关键帧条件生成的新图**，并带溯源元信息 `(keyframe_ts_i, bbox_i, source_object o_i)`。

**形式化目标**：学习映射 `f: (V, Q) → A`，使
1. **文本正确**：`t_*` 正确回答 `Q`；
2. **图文一致**：`img_i` 与其相邻文本 `t_i` 语义一致；
3. **关键帧一致（核心）**：`img_i` 忠实于其溯源关键帧 `V[keyframe_ts_i]` 中的目标 `o_i`（主体/外观可识别地保持）；
4. **定位正确**：`(keyframe_ts_i, bbox_i)` 确实是 `Q` 所指内容在 `V` 中的正确时空位置。

**四个区分性特征**（缺一不可，正是 novelty 交点）：(a) 视频输入 · (b) 输出**生成**的新图 · (c) 图文交错 · (d) MAS+多模态RAG 编排。

---

## 2. 专属基准：VKIG-Bench（暂名）
**V**ideo-**K**eyframe-grounded **I**nterleaved **G**eneration Benchmark

### 2.1 每条样本结构
```json
{
  "video": "clip_0001.mp4",
  "query": "主角第一次见到那辆红色跑车时，车停在哪、什么样？",
  "gold": {
    "text": ["主角在剧院门口第一次见到红色跑车，", "它停在台阶右侧。"],
    "evidence": [
      {"keyframe_ts": 87.3, "bbox": [x1,y1,x2,y2], "object": "red sports car",
       "gold_image_ref": "kf_0001_car.png", "pseudo_bbox": false}
    ]
  },
  "meta": {"source": "vstar|self_drama", "split": "test"}
}
```

### 2.2 数据来源（三路，先少后多）
1. **V-STaR 种子**（公开、自带 bbox + when/where/what）→ 直接转成上述 schema，作为有标注主干。
2. **自建影视/短剧**（待 Q1 确认）→ `video-subtitle-extractor`(字幕) + `DeepSeek-OCR`(画面文字) + Grounding-DINO(伪框)，半自动造样本 + 人工抽检。
3. **缺框兜底**：无 bbox 来源用 Grounding-DINO 按 `Q` 中实体跑伪框，标 `pseudo_bbox=true`，人工校验子集。

### 2.3 规模与划分（起步）
- 起步 **300–500 条**（test 为主，可投版本够用）；后续扩到 1–2k。
- 划分：dev 100 / test 300+；每条至少 1 个 evidence。

### 2.4 评测维度（= 护城河）
| 维度 | 指标 | 说明 | 是否现有基准缺失 |
|---|---|---|---|
| 文本正确 | GPT-4o-judge / 准确率 | 答案对不对 | 否 |
| 图文一致 | **VQAScore** | 生成图是否匹配相邻文本/指令 | 否 |
| **关键帧一致** | **DreamSim / DINO 相似度**（生成图 vs 溯源关键帧目标） | **我们独有维度**——非视频基准无法评 | **✅ 是(核心卖点)** |
| **时空定位** | 时间 IoU(ts) + 空间 AP/IoU(bbox) | 图是否对应正确时刻/物体 | **✅ 是** |
| 画质 | FID/KID | 生成图真实感 | 否 |
| 总体 | 人工胜率(A/B) | 终评 | 否 |

> **与现有基准的差异**：MRAMG-Bench / FIG-Eval 评"图文交错/事实图生成"但**无视频、无关键帧溯源一致性**；V-STaR 有视频+bbox 但评的是 QA 不评"生成图"。VKIG-Bench = **视频溯源 + 生成图 + 关键帧一致性**三者合一，填补空白。

### 2.5 构造流水线
```
原始视频 ─► 关键帧抽取(Q-driven, #2) ─► 目标 grounding(框/分割, #1 GROVE/G-DINO)
        ─► gold image: 从关键帧目标"裁块作参考"→(可选)用生成模型产 1 张参考级 gold
        ─► 人工抽检(框准不准/图对不对) ─► 入库(schema 2.1)
```

---

## 3. 待办（issue #11 跟踪）
- [ ] 老师确认 Q1 场景 → 锁定数据来源 2
- [ ] 拉 V-STaR，写 `vstar → VKIG schema` 转换脚本（替代/扩展 `mvp_code/prepare_*`）
- [ ] 定 gold image 生成口径（裁块参考 vs 模型生成参考）
- [ ] 实现 4 个评测指标脚本（VQAScore / DreamSim / 时空IoU / FID）
- [ ] 先出 50 条 dev 跑通评测闭环，再扩到 300+

> 待老师确认项：场景(Q1)、是否建基准(Q2)、命名是否合适。确认后本草案升级为正式 benchmark spec。
