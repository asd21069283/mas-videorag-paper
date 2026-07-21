# confusion_kappa.py — 人评×MLLM裁判 混淆矩阵 + Cohen κ(§6.7 表格配套, 纯标准库)
# 宽松档(Round 1, n=39 排除1例unsure) 与 约束档(Round 3 D档, n=40)
import json, os

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def confusion(labels, judge):
    tp = tn = fp = fn = exc = 0
    for vid, h in labels.items():
        j = (judge.get(vid) or {}).get("judge") or (judge.get(vid) or {}).get("same_moment")
        if h == "unsure":
            exc += 1; continue
        if h == "yes" and j == "yes": tp += 1
        elif h == "no" and j == "no": tn += 1
        elif h == "no" and j == "yes": fp += 1
        elif h == "yes" and j == "no": fn += 1
    n = tp + tn + fp + fn
    po = (tp + tn) / n
    ph, pj = (tp + fn) / n, (tp + fp) / n
    pe = ph * pj + (1 - ph) * (1 - pj)
    kappa = (po - pe) / (1 - pe)
    return {"TP": tp, "FN": fn, "FP": fp, "TN": tn, "n": n, "excluded_unsure": exc,
            "agreement": round(po, 4), "cohen_kappa": round(kappa, 4)}


def main():
    r1 = confusion(json.load(open(f"{ROOT}/demo_2026-07-14/human_faithfulness_labels.json")),
                   json.load(open(f"{ROOT}/demo_2026-07-14/mllm_judge_40.json")))
    d = json.load(open(f"{ROOT}/demo_2026-07-18/dual_axis_labels_D40.json"))
    r3 = confusion({v: x["faithful"] for v, x in d.items()},
                   json.load(open(f"{ROOT}/demo_2026-07-18/mllm_judge_d40.json")))
    out = {"round1_permissive": r1, "round3_constrained_D": r3}
    print(json.dumps(out, indent=2))
    json.dump(out, open(f"{ROOT}/demo_2026-07-18/confusion_kappa.json", "w"), indent=2)
    print("->", f"{ROOT}/demo_2026-07-18/confusion_kappa.json")


if __name__ == "__main__":
    main()
