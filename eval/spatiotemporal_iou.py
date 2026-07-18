# spatiotemporal_iou.py
# ⚠️ 二级/独立校验器, 非论文数字的权威来源(2026-07-18 代码审计标注)。
#   论文 §6.3–6.6 的所有已发表定位数字由 pipeline/hcstvg_eval.py 与 batch_eval.py 产出。
#   本脚本与它们【时间口径一致】(传 --t_pad 0 即严格命中), 但【空间 gold 口径不同】:
#   本脚本对 gold evidence[0] 的固定框打分, 而 hcstvg_eval 取"距预测关键帧时刻最近的 tube 框"。
#   因此两者在 temporal-hit 门控上一致、但逐样本 spatial IoU 不应期望逐位相等——勿把本脚本当复现基线。
#   已知局限(不影响任何已发表数字, 仅防未来误用): evaluate() 对 pred 的 video 键在 gold 里找不到时
#   静默 continue(缩小分母); gold_by 按 video 键折叠, 同一 video 多条样本只保留最后一条;
#   若 pred 行把数据集 gold 副本嵌在 "gold" 键下会被当预测自比。多 QA/enriched-dump 场景勿直接用。
# VKIG 评测·"时空定位"维度(纯 Python, 无需 GPU/模型, 可直接跑)。
# 衡量: 预测的 evidence(keyframe_ts + bbox) 是否对上 gold 的时刻与位置。
#   - temporal IoU: 预测关键帧时刻 vs gold 时间区间 [start,end] 的命中/IoU
#   - spatial IoU : 预测框 vs gold 关键帧框
#   - m-vIoU      : 同时满足时间命中且空间 IoU>=阈值 才算定位正确(论文常用 m_tIoU/m_vIoU 思路)
# 用法:
#   python spatiotemporal_iou.py --pred pred.jsonl --gold vkig.jsonl --siou_thr 0.5
#   python spatiotemporal_iou.py --selftest
import argparse, json


def iou_1d(a, b):
    """两个时间区间 [s,e] 的 IoU。"""
    s = max(a[0], b[0]); e = min(a[1], b[1])
    inter = max(0.0, e - s)
    union = (a[1] - a[0]) + (b[1] - b[0]) - inter
    return inter / union if union > 0 else 0.0


def iou_2d(a, b):
    """两个框 [x1,y1,x2,y2] 的 IoU。"""
    x1 = max(a[0], b[0]); y1 = max(a[1], b[1])
    x2 = min(a[2], b[2]); y2 = min(a[3], b[3])
    iw = max(0.0, x2 - x1); ih = max(0.0, y2 - y1)
    inter = iw * ih
    area_a = max(0.0, a[2] - a[0]) * max(0.0, a[3] - a[1])
    area_b = max(0.0, b[2] - b[0]) * max(0.0, b[3] - b[1])
    union = area_a + area_b - inter
    return inter / union if union > 0 else 0.0


def temporal_hit(pred_ts, gold_interval, pad=0.0):
    """预测关键帧时刻是否落在 gold 区间内(可加 pad 容差秒)。"""
    if gold_interval is None or pred_ts is None:
        return False
    return (gold_interval[0] - pad) <= pred_ts <= (gold_interval[1] + pad)


def score_sample(pred_ev, gold_ev, siou_thr=0.5, t_pad=0.5):
    """单条 evidence 配对打分。pred_ev/gold_ev: {keyframe_ts, bbox, interval}。
    返回 dict: t_hit, s_iou, correct(时间命中且空间IoU>=阈值)。"""
    t_hit = temporal_hit(pred_ev.get("keyframe_ts"), gold_ev.get("interval"), pad=t_pad)
    s_iou = 0.0
    if pred_ev.get("bbox") and gold_ev.get("bbox"):
        s_iou = iou_2d(pred_ev["bbox"], gold_ev["bbox"])
    return {"t_hit": bool(t_hit), "s_iou": s_iou, "correct": bool(t_hit and s_iou >= siou_thr)}


def evaluate(pred_rows, gold_rows, siou_thr=0.5, t_pad=0.5):
    """按顺序(或可改成按 id 对齐)配对首条 evidence, 汇总。"""
    gold_by = {g.get("video"): g for g in gold_rows}
    n, t_hits, s_ious, corrects = 0, 0, [], 0
    for p in pred_rows:
        g = gold_by.get(p.get("video"))
        if not g:
            continue
        pev = (p.get("gold") or p).get("evidence", [{}])
        gev = g["gold"]["evidence"]
        if not gev:
            continue
        r = score_sample(pev[0] if pev else {}, gev[0], siou_thr, t_pad)
        n += 1; t_hits += r["t_hit"]; s_ious.append(r["s_iou"]); corrects += r["correct"]
    return {
        "n": n,
        "temporal_recall": t_hits / n if n else 0.0,
        "mean_spatial_iou": sum(s_ious) / len(s_ious) if s_ious else 0.0,
        "localization_acc@%.2f" % siou_thr: corrects / n if n else 0.0,
    }


def _load(path):
    txt = open(path, encoding="utf-8").read().strip()
    if not txt:
        return []
    return json.loads(txt) if txt[0] == "[" else [json.loads(l) for l in txt.splitlines() if l.strip()]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pred", required=True); ap.add_argument("--gold", required=True)
    ap.add_argument("--siou_thr", type=float, default=0.5)
    ap.add_argument("--t_pad", type=float, default=0.5)
    a = ap.parse_args()
    res = evaluate(_load(a.pred), _load(a.gold), a.siou_thr, a.t_pad)
    print(json.dumps(res, ensure_ascii=False, indent=2))


def selftest():
    assert abs(iou_1d([0, 10], [5, 15]) - (5 / 15)) < 1e-9
    assert iou_1d([0, 5], [10, 20]) == 0.0
    assert abs(iou_2d([0, 0, 10, 10], [0, 0, 10, 10]) - 1.0) < 1e-9
    assert abs(iou_2d([0, 0, 2, 2], [1, 1, 3, 3]) - (1.0 / 7.0)) < 1e-9   # inter1 / union7
    assert iou_2d([0, 0, 1, 1], [5, 5, 6, 6]) == 0.0
    assert temporal_hit(4.0, [2.0, 6.0]) is True
    assert temporal_hit(8.0, [2.0, 6.0]) is False
    assert temporal_hit(6.4, [2.0, 6.0], pad=0.5) is True

    gold = [{"video": "v1", "gold": {"evidence": [{"keyframe_ts": 4.0, "interval": [2.0, 6.0], "bbox": [100, 50, 200, 300]}]}}]
    pred_good = [{"video": "v1", "gold": {"evidence": [{"keyframe_ts": 4.2, "bbox": [105, 55, 205, 305]}]}}]
    pred_bad = [{"video": "v1", "gold": {"evidence": [{"keyframe_ts": 12.0, "bbox": [0, 0, 5, 5]}]}}]
    rg = evaluate(pred_good, gold, siou_thr=0.5)
    assert rg["n"] == 1 and rg["temporal_recall"] == 1.0 and rg["mean_spatial_iou"] > 0.8 and rg["localization_acc@0.50"] == 1.0, rg
    rb = evaluate(pred_bad, gold, siou_thr=0.5)
    assert rb["temporal_recall"] == 0.0 and rb["localization_acc@0.50"] == 0.0, rb
    print("✅ spatiotemporal_iou selftest 通过: 1D/2D IoU / 时间命中 / 定位准确率聚合")


if __name__ == "__main__":
    import sys
    selftest() if "--selftest" in sys.argv else main()
