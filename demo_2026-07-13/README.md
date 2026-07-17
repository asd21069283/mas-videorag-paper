# demo_2026-07-13 数据说明(审计备注 2026-07-17)

- **sel_e2_300 / sel_e2_1000 / sel_e2_g15 / sel_e2_g20 的 `summary.params` 字段**:
  这些 E2 run 的 `params` 里写的 `w_obj: 0.4, w_act: 0.3` 是当时 CLI 参数的**未使用默认值**。
  E2 的 Stage B 实际生效配置为**两通道、`w_obj = 0.6` 写死**(见 `pipeline/hcstvg_selector_v2.py`,
  e2 分支 `combine_scores(..., w_obj=0.6)`);动作通道仅 E1 使用(E1 生效权重 w_q/w_obj/w_act = 0.3/0.4/0.3)。
  E1 与 E2 是两个独立改动,**未叠加**(全因子消融未做)。脚本已修复为记录生效值;本目录历史 json 保持原样不改。
- **baseline_FIX100_dump.json**: 中间产物,已被 `baseline_FINAL300_dump.json` 取代(superseded),仅留作复现审计线索。
- `compare_FINAL300.json` 同时含 `baseline_bridged_floor` 与 `baseline_bridged_ceil` 两档(sted 中点取整敏感性双报)
  及 `baseline_native`(原生口径复现);论文 §6.6 表报 floor 档,双档数字均已在论文桥接协议段披露。
