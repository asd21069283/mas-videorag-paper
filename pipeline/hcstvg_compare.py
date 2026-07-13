# hcstvg_compare.py — 同clip对比: NeurIPS'25 baseline(dump+位置映射) vs 我们(main1000)
# 用法: python pipeline/hcstvg_compare.py --dump <dump_pred.json> --pos <pos2vid.json> --ours <hcstvg_main1000.json>
import json, re, argparse

def parse_box(b):
    while isinstance(b, list) and len(b) == 1 and isinstance(b[0], (list, str)):
        b = b[0]
    if isinstance(b, str):
        n = re.findall(r'-?\d+\.?\d*', b); return [float(x) for x in n[:4]]
    return [float(x) for x in b[:4]]

def iou(a, b):
    a = parse_box(a); b = parse_box(b)
    x1 = max(a[0], b[0]); y1 = max(a[1], b[1]); x2 = min(a[2], b[2]); y2 = min(a[3], b[3])
    inter = max(0, x2-x1) * max(0, y2-y1)
    ua = (a[2]-a[0])*(a[3]-a[1]) + (b[2]-b[0])*(b[3]-b[1]) - inter
    return inter/ua if ua > 0 else 0.0

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dump", required=True); ap.add_argument("--pos", required=True)
    ap.add_argument("--ours", required=True); ap.add_argument("--out", default="")
    a = ap.parse_args()
    bl = json.load(open(a.dump)); pos = json.load(open(a.pos))
    ours = {r["vid"]: r for r in json.load(open(a.ours))["per_sample"]}
    bl_rows = {}
    for k, tube in bl["predictions"].items():
        stem = pos.get(k, {}).get("stem")
        if not stem or k not in bl["video_predictions"] or k not in bl["gt_box"]:
            continue
        gt = bl["gt_box"][k]; gtf = sorted(int(f) for f in gt)
        sted = bl["video_predictions"][k]["sted"]; kf = (int(sted[0])+int(sted[1]))//2
        pf = sorted(int(f) for f in tube); pk = min(pf, key=lambda f: abs(f-kf))
        gk = min(gtf, key=lambda f: abs(f-kf))
        bl_rows[stem] = {"temporal_hit": bool(gtf[0] <= kf <= gtf[-1]),
                         "spatial_iou": iou(tube[str(pk)], gt[str(gk)])}
    common = [s for s in bl_rows if s in ours]
    def summ(rows):
        n = len(rows) or 1
        hit = [r["spatial_iou"] for r in rows if r.get("temporal_hit") and "spatial_iou" in r]
        return {"n": n,
                "temporal_recall": round(sum(1 for r in rows if r.get("temporal_hit"))/n, 4),
                "acc@0.3_JOINT": round(sum(1 for r in rows if r.get("temporal_hit") and r.get("spatial_iou", 0) >= 0.3)/n, 4),
                "acc@0.5_JOINT": round(sum(1 for r in rows if r.get("temporal_hit") and r.get("spatial_iou", 0) >= 0.5)/n, 4),
                "iou_when_hit": round(sum(hit)/len(hit), 4) if hit else None}
    result = {"n_common": len(common),
              "baseline_bridged": summ([bl_rows[s] for s in common]),
              "ours": summ([ours[s] for s in common]),
              "baseline_native": {k: round(v, 4) for k, v in bl["metrics"].items() if not isinstance(v, dict)}}
    print(json.dumps(result, ensure_ascii=False, indent=2))
    if a.out:
        json.dump(result, open(a.out, "w"), ensure_ascii=False, indent=2); print("[saved]", a.out)

if __name__ == "__main__":
    main()
