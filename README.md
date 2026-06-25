# MAS-VideoRAG: Video+Text → Image+Text

一个**多智能体（MAS）+ 多模态 RAG** 的视频系统研究项目。核心任务：

> **输入「视频 + 文字指令」→ 输出「图片 + 文字」**

- **Understand Agent**：理解视频 + 检索知识，定位关键证据（关键帧/片段）。
- **Generation Agent**：据此对视频内目标定位、裁剪或生成图像，配文输出。
- 两个 Agent 共享一个**多模态 RAG** 知识库。

填补的空白（模态组合）：VideoRAG = 视频+文字→文字；M2RAG = 图+文字→图+文字；ViMax = 图→视频；**本项目 = 视频+文字→图+文字**。

---

## 仓库结构

```
.
├── docs_文献调研与进展.md      # 文献综述(15+篇)/3个创新点/定题/实验计划
├── 创新点2_方法论框架.md        # 主线方法: Q-driven 客体级定位→裁剪/生成统一链路(含架构图/公式/消融)
└── mvp_code/                   # MVP 阶段一: 视频+文字→文字 (Qwen3-VL-4B + MUSIC-AVQA + ms-swift LoRA)
    ├── README_MVP.md           # AutoDL 从零跑通步骤
    ├── requirements.txt
    ├── prepare_musicavqa.py    # MUSIC-AVQA → ms-swift 训练格式
    ├── train_lora.sh           # LoRA(冻结ViT) 微调
    ├── infer.py
    └── merge_and_eval.py
```

## 路线图（Roadmap）

- [ ] MVP：视频+文字→文字 理解链跑通（Qwen3-VL-4B + MUSIC-AVQA）
- [ ] 接入 Q-driven 客体级关键帧选择（AKS 升级）+ Grounding-DINO 定位
- [ ] 裁剪/生成决策闭环（M2RAG 输出端基线对照）
- [ ] 双 Agent 闭环 + 共享多模态 RAG
- [ ] 图文互证去幻觉（可信 RAG）
- [ ] 评测：M2RAG 8 指标 + GPT-4o-judge + CUVA 因果维度 + 人工胜率

## Baselines
- 理解侧：[VideoRAG](https://github.com/HKUDS/VideoRAG)
- 输出侧：[M2RAG](https://github.com/maziao/M2RAG)（arXiv 2411.16365）

> 进展、想法、待办都走本仓 Issues 跟踪。
