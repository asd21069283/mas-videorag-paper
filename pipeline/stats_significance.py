# stats_significance.py — 定位侧数值评估: Bootstrap 95% CI + 配对显著性(McNemar/Wilcoxon)
# 纯标准库(无 numpy/scipy)。复现上报点估计, 再加误差棒与显著性。
# 用法: python pipeline/stats_significance.py
import json, os, math, random

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
random.seed(20260720)  # 固定种子, 结果可复现

FILES = {
    "E0@300 (uniform)": "demo_2026-07-04/results_json/hcstvg_bl_uniform.json",
    "random@300":       "demo_2026-07-04/results_json/hcstvg_bl_random.json",
    "oracle@300":       "demo_2026-07-04/results_json/hcstvg_bl_middle_gt.json",
    "E1@300":           "demo_2026-07-13/sel_e1_300.json",
    "E2@300 (g10)":     "demo_2026-07-13/sel_e2_300.json",
    "E0@1000 (main)":   "demo_2026-07-04/results_json/hcstvg_main1000.json",
    "E2@1000":          "demo_2026-07-13/sel_e2_1000.json",
    "E2 g15":           "demo_2026-07-18/sel_e2_g15_fix.json",
    "E2 g20":           "demo_2026-07-18/sel_e2_g20_fix.json",
}


def load(f):
    return json.load(open(os.path.join(ROOT, f)))["per_sample"]


def rate(rows, pred):
    return sum(1 for r in rows if pred(r)) / len(rows)


def acc03(r): return bool(r.get("temporal_hit")) and r.get("spatial_iou", 0) >= 0.3
def acc05(r): return bool(r.get("temporal_hit")) and r.get("spatial_iou", 0) >= 0.5
def trec(r):  return bool(r.get("temporal_hit"))


def bootstrap_ci(rows, pred, B=10000):
    n = len(rows); vals = [1 if pred(r) else 0 for r in rows]
    boots = []
    for _ in range(B):
        s = sum(vals[random.randrange(n)] for _ in range(n))
        boots.append(s / n)
    boots.sort()
    return boots[int(0.025 * B)], boots[int(0.975 * B)]


def mcnemar(rows_a, rows_b, pred):
    # 配对(同序): b = a对b错, c = a错b对; 精确二项双侧
    b = c = 0
    for ra, rb in zip(rows_a, rows_b):
        pa, pb = pred(ra), pred(rb)
        if pa and not pb: b += 1
        elif pb and not pa: c += 1
    n = b + c
    if n == 0: return b, c, 1.0
    k = min(b, c)
    p = sum(math.comb(n, i) for i in range(0, k + 1)) / (2 ** n) * 2
    return b, c, min(1.0, p)


def wilcoxon_iou(rows_a, rows_b):
    # spatial_iou 配对差(缺失当0)的 signed-rank, 正态近似
    diffs = []
    for ra, rb in zip(rows_a, rows_b):
        d = rb.get("spatial_iou", 0) - ra.get("spatial_iou", 0)
        if d != 0: diffs.append(d)
    if not diffs: return 0.0, 1.0
    order = sorted(range(len(diffs)), key=lambda i: abs(diffs[i]))
    ranks = [0] * len(diffs); i = 0
    while i < len(order):
        j = i
        while j < len(order) and abs(diffs[order[j]]) == abs(diffs[order[i]]): j += 1
        r = (i + 1 + j) / 2
        for k in range(i, j): ranks[order[k]] = r
        i = j
    Wp = sum(ranks[i] for i in range(len(diffs)) if diffs[i] > 0)
    n = len(diffs); mu = n * (n + 1) / 4; sig = math.sqrt(n * (n + 1) * (2 * n + 1) / 24)
    z = (Wp - mu) / sig
    return z, math.erfc(abs(z) / math.sqrt(2))


def main():
    data = {k: load(v) for k, v in FILES.items()}
    out = {"point_and_ci": {}, "paired_tests": {}}
    print("=== Bootstrap 95% CI (10k) ===")
    for name, rows in data.items():
        row = {}
        for mk, pred in [("temporal_recall", trec), ("acc@0.3", acc03), ("acc@0.5", acc05)]:
            pt = rate(rows, pred); lo, hi = bootstrap_ci(rows, pred)
            row[mk] = [round(pt, 4), round(lo, 4), round(hi, 4)]
        out["point_and_ci"][name] = row
        print(f"  {name:20s} acc@0.3={row['acc@0.3'][0]:.3f} [{row['acc@0.3'][1]:.3f},{row['acc@0.3'][2]:.3f}]  "
              f"trec={row['temporal_recall'][0]:.3f}  acc@0.5={row['acc@0.5'][0]:.3f}")

    print("\n=== Paired significance (McNemar on acc@0.3 + Wilcoxon on spatial_iou) ===")
    PAIRS = [
        ("E0@300 (uniform)", "E2@300 (g10)"),
        ("E1@300", "E2@300 (g10)"),
        ("E0@1000 (main)", "E2@1000"),
        ("E2@300 (g10)", "E2 g20"),
    ]
    for a, b in PAIRS:
        bb, cc, p = mcnemar(data[a], data[b], acc03)
        z, wp = wilcoxon_iou(data[a], data[b])
        aa = rate(data[a], acc03); ab = rate(data[b], acc03)
        out["paired_tests"][f"{a} → {b}"] = {
            "acc@0.3": [round(aa, 4), round(ab, 4)], "delta": round(ab - aa, 4),
            "mcnemar_b_c": [bb, cc], "mcnemar_p": round(p, 5),
            "wilcoxon_z": round(z, 3), "wilcoxon_p": round(wp, 5),
        }
        sig = "***" if p < 0.001 else "**" if p < 0.01 else "*" if p < 0.05 else "n.s."
        print(f"  {a} → {b}: acc {aa:.3f}→{ab:.3f} (Δ{ab-aa:+.3f})  "
              f"McNemar b/c={bb}/{cc} p={p:.4f} {sig}  Wilcoxon(IoU) p={wp:.4f}")

    json.dump(out, open(os.path.join(ROOT, "demo_2026-07-18/stats_significance.json"), "w"),
              ensure_ascii=False, indent=2)
    print("\n->", "demo_2026-07-18/stats_significance.json")


if __name__ == "__main__":
    main()
