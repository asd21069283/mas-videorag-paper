# hcstvg_bridge.py
# 把 NeurIPS'25 baseline(LLaVA_Next_STVG) 的 dump_pred.json 换算成【我们的关键帧联合口径】,
# 使其与我们的方法在同一指标下可比(答老师Q3的口径桥接)。
# baseline 输出整条时空管(逐帧box + 预测时段 sted[帧]); 我们取【预测时段中点帧+该帧box】当"单关键帧预测",
# 再按我们的 联合定位(temporal_hit ∧ spatial_iou>=τ) 打分, 用 baseline 自带的 gt_box(同HC-STVG gold)。
# 用法: python hcstvg_bridge.py --dump dump_pred.json
import json, argparse


import re as _re


def parse_box(box):
    """兼容3种: 真list[x1,y1,x2,y2] / 嵌套[[...]] / numpy repr字符串'[377 127 438 323]'
    (dump用了default=str, np.array被字符串化)。返回 [x1,y1,x2,y2] float。"""
    while isinstance(box, list) and len(box) == 1 and isinstance(box[0], (list, str)):
        box = box[0]
    if isinstance(box, str):
        nums = _re.findall(r'-?\d+\.?\d*', box)
        return [float(x) for x in nums[:4]]
    return [float(x) for x in box[:4]]


def iou(a, b):
    a = parse_box(a); b = parse_box(b)
    if len(a) < 4 or len(b) < 4:
        return 0.0
    x1 = max(a[0], b[0]); y1 = max(a[1], b[1]); x2 = min(a[2], b[2]); y2 = min(a[3], b[3])
    inter = max(0.0, x2 - x1) * max(0.0, y2 - y1)
    ua = (a[2]-a[0])*(a[3]-a[1]) + (b[2]-b[0])*(b[3]-b[1]) - inter
    return inter / ua if ua > 0 else 0.0


def unwrap(box):
    return box  # parse_box 里已处理嵌套/字符串


def bridge(dump):
    R = []
    preds = dump["predictions"]; vpreds = dump["video_predictions"]; gts = dump.get("gt_box", {})
    for vid, tube in preds.items():
        if vid not in gts or vid not in vpreds:
            continue
        gt = gts[vid]                                     # {frame: [[box]]}
        gt_frames = sorted(int(f) for f in gt)
        if not gt_frames or not tube:
            continue
        gold_s, gold_e = gt_frames[0], gt_frames[-1]      # gold 时段(帧)
        sted = vpreds[vid].get("sted", [0, 0])            # 预测时段(帧)
        kf = (int(sted[0]) + int(sted[1])) // 2           # 预测关键帧 = 时段中点
        # 时间命中: 关键帧落在 gold 时段(严格, 与我们口径一致)
        thit = bool(gold_s <= kf <= gold_e)
        # 预测框: 取预测帧里最接近 kf 的
        pframes = sorted(int(f) for f in tube)
        pk = min(pframes, key=lambda f: abs(f - kf))
        pbox = unwrap(tube[str(pk)])
        # gold框: 取 gold 帧里最接近 kf 的
        gk = min(gt_frames, key=lambda f: abs(f - kf))
        gbox = unwrap(gt[str(gk)])
        siou = round(iou(pbox, gbox), 4)
        R.append({"vid": vid, "temporal_hit": thit, "spatial_iou": siou, "kf": kf,
                  "gold_window": [gold_s, gold_e], "pred_sted": [int(sted[0]), int(sted[1])]})
    N = len(R) or 1
    def ja(t): return round(sum(1 for r in R if r["temporal_hit"] and r["spatial_iou"] >= t)/N, 4)
    hit = [r["spatial_iou"] for r in R if r["temporal_hit"]]
    summary = {
        "method": "NeurIPS25-ZeroShotSTVG (bridged to our keyframe metric)",
        "n": len(R),
        "temporal_recall": round(sum(1 for r in R if r["temporal_hit"])/N, 4),
        "localization_acc@0.3_JOINT": ja(0.3),
        "localization_acc@0.5_JOINT": ja(0.5),
        "spatial_iou_when_temporal_hit": round(sum(hit)/len(hit), 4) if hit else None,
        # 附: baseline 自家原生指标(直接读 dump.metrics)
        "native_m_vIoU": round(dump.get("metrics", {}).get("viou", 0), 4),
        "native_viou@0.3": round(dump.get("metrics", {}).get("viou@0.3", 0), 4),
        "native_viou@0.5": round(dump.get("metrics", {}).get("viou@0.5", 0), 4),
    }
    return summary, R


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dump", required=True)
    ap.add_argument("--out", default="")
    a = ap.parse_args()
    dump = json.load(open(a.dump, encoding="utf-8"))
    summary, R = bridge(dump)
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    if a.out:
        json.dump({"summary": summary, "per_sample": R}, open(a.out, "w", encoding="utf-8"),
                  ensure_ascii=False, indent=2)
        print("[saved]", a.out)


if __name__ == "__main__":
    main()
