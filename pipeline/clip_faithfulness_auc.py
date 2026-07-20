# clip_faithfulness_auc.py — 量化"CLIP 相似度作为忠实性代理"的判别力(§6.7 数值评估)
# 结论: CLIP subject sim 与人评忠实性相关但不足以替代协议——
#   宽松档 AUC 0.85 却无可靠阈值(72% 重叠); 约束档 AUC 降至 0.67(失败变细→CLIP 崩塌)。
# 用法: python pipeline/clip_faithfulness_auc.py   (纯本机, 无 GPU)
import json, math, statistics as st, os

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def auc(pos, neg):
    w = sum(1 for a in pos for b in neg if a > b) + 0.5 * sum(1 for a in pos for b in neg if a == b)
    return w / (len(pos) * len(neg)) if pos and neg else float("nan")


def mann_whitney_p(pos, neg):
    allv = sorted([(c, 1) for c in pos] + [(c, 0) for c in neg])
    ranks = {}; i = 0
    while i < len(allv):
        j = i
        while j < len(allv) and allv[j][0] == allv[i][0]:
            j += 1
        r = (i + 1 + j) / 2
        for k in range(i, j):
            ranks[k] = r
        i = j
    R1 = sum(ranks[idx] for idx, (c, g) in enumerate(allv) if g == 1)
    n1, n2 = len(pos), len(neg)
    U1 = R1 - n1 * (n1 + 1) / 2
    mu = n1 * n2 / 2; sig = math.sqrt(n1 * n2 * (n1 + n2 + 1) / 12)
    z = (U1 - mu) / sig
    return z, math.erfc(abs(z) / math.sqrt(2))


def best_threshold(pos, neg):
    best = None
    for ti in range(300, 950):
        t = ti / 1000
        tp = sum(1 for c in pos if c >= t); fp = sum(1 for c in neg if c >= t)
        acc = (tp + (len(neg) - fp)) / (len(pos) + len(neg))
        if best is None or acc > best[1]:
            best = (t, acc, len(pos) - tp, fp)
    return best  # threshold, acc, missed_faithful, false_accept


def analyze(name, pairs):
    pos = [c for c, l in pairs if l == "yes"]; neg = [c for c, l in pairs if l == "no"]
    a = auc(pos, neg); z, p = mann_whitney_p(pos, neg)
    t, acc, fn, fp = best_threshold(pos, neg)
    lo, hi = max(min(pos), min(neg)), min(max(pos), max(neg))
    overlap = sum(1 for c in pos + neg if lo <= c <= hi)
    print(f"\n=== {name} (n={len(pairs)}) ===")
    print(f"  忠实组 mean={st.mean(pos):.3f} (n={len(pos)}) | 不忠实组 mean={st.mean(neg):.3f} (n={len(neg)})")
    print(f"  AUC={a:.3f}  Mann-Whitney z={z:.2f} p={p:.3f}")
    print(f"  最佳单阈值={t:.3f} acc={acc:.1%} (仍漏判忠实{fn} + 误纳不忠实{fp})")
    print(f"  重叠区[{lo:.3f},{hi:.3f}] 含 {overlap}/{len(pairs)}={overlap/len(pairs):.0%} 样本(阈值无法判定)")
    return {"n": len(pairs), "auc": round(a, 3), "p": round(p, 3), "best_thr": round(t, 3),
            "best_acc": round(acc, 3), "overlap_frac": round(overlap / len(pairs), 3)}


def main():
    out = {}
    # Round 1 宽松版
    gen = json.load(open(f"{ROOT}/demo_2026-07-14/gen_e2_40_results.json"))
    hf = json.load(open(f"{ROOT}/demo_2026-07-14/human_faithfulness_labels.json"))
    per = {r["vid"]: r for r in gen["per_sample"]}
    out["permissive"] = analyze("Round1 宽松版 (CLIP subject sim)",
                                 [(per[v]["clip_subject_sim"], l) for v, l in hf.items() if v in per and l in ("yes", "no")])
    # D 档约束版
    cd = json.load(open(f"{ROOT}/demo_2026-07-18/clip_d40_results.json"))["per_sample"]
    dl = json.load(open(f"{ROOT}/demo_2026-07-18/dual_axis_labels_D40.json"))
    out["constrained_D"] = analyze("Round3 D档约束版 (CLIP subject sim)",
                                   [(cd[v], dl[v]["faithful"]) for v in cd if v in dl and dl[v]["faithful"] in ("yes", "no")])
    json.dump(out, open(f"{ROOT}/demo_2026-07-18/clip_faithfulness_auc.json", "w"), ensure_ascii=False, indent=2)
    print("\n->", f"{ROOT}/demo_2026-07-18/clip_faithfulness_auc.json")


if __name__ == "__main__":
    main()
