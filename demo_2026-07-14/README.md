# demo_2026-07-14 数据说明(审计备注 2026-07-17)

- **mllm_judge_ab10.json 键 `152_5LrOQEt_XVM`**: 该条的 `raw` 字段是被截断的非法 JSON(缺右花括号),
  从 raw 重放解析会失败。**`judge` 字段为权威**(该条 judge='yes',与截断内容语义一致,已正确计入
  强约束 3/10)。其余 49 条(judge_40 全部 + ab10 另 9 条)raw 均可解析且与 judge 字段一致。
- 人评标签(human_faithfulness_labels / human_labels_ab10)为单一作者标注,无标注者间一致性;
  论文 §7 已声明,camera-ready 前需多标注者复标。
