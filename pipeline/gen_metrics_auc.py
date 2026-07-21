# gen_metrics_auc.py — 自动指标家族(DreamSim/DINO/CLIP/LPIPS)对人评忠实性的判别力(两档AUC)
# 指标值由实例侧 gen_metrics_fast.py / gen_metrics_ds.py / clip 系脚本产出(demo_2026-07-21, demo_2026-07-14/18)
# 本脚本: join人评标签 → 每指标每档算AUC(方向统一为"越大越忠实"; 距离类取负)
import json, os

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def auc(pos, neg):
    w = sum(1 for a in pos for b in neg if a > b) + 0.5 * sum(1 for a in pos for b in neg if a == b)
    return w / (len(pos) * len(neg)) if pos and neg else float("nan")


def main():
    hf = json.load(open(f"{ROOT}/demo_2026-07-14/human_faithfulness_labels.json"))
    dl = json.load(open(f"{ROOT}/demo_2026-07-18/dual_axis_labels_D40.json"))
    labels = {
        "e2_40": {v: l for v, l in hf.items() if l in ("yes", "no")},
        "d40": {v: d["faithful"] for v, d in dl.items() if d["faithful"] in ("yes", "no")},
    }
    # 指标源: (文件模板, 键, 越大越忠实?)
    srcs = [
        ("dreamsim", "demo_2026-07-21/gen_metrics_ds_{tag}.json", "dreamsim", False),
        ("dino_sim", "demo_2026-07-21/gen_metrics_fast_{tag}.json", "dino_sim", True),
        ("lpips", "demo_2026-07-21/gen_metrics_fast_{tag}.json", "lpips", False),
    ]
    clip_src = {"e2_40": ("demo_2026-07-14/gen_e2_40_results.json", "clip_subject_sim"),
                "d40": ("demo_2026-07-18/clip_d40_results.json", None)}

    out = {}
    for name, tpl, key, hb in srcs:
        row = {}
        for tag in ("e2_40", "d40"):
            m = json.load(open(f"{ROOT}/{tpl.format(tag=tag)}"))["per_sample"]
            sgn = 1 if hb else -1
            pos = [sgn * m[v][key] for v, l in labels[tag].items() if l == "yes" and v in m]
            neg = [sgn * m[v][key] for v, l in labels[tag].items() if l == "no" and v in m]
            row[tag] = {"auc": round(auc(pos, neg), 3),
                        "faithful_mean": round(sgn * sum(pos) / len(pos), 4),
                        "unfaithful_mean": round(sgn * sum(neg) / len(neg), 4)}
        out[name] = row
    # CLIP(两档来源不同格式)
    row = {}
    g = json.load(open(f"{ROOT}/{clip_src['e2_40'][0]}"))["per_sample"]
    m = {r["vid"]: r["clip_subject_sim"] for r in g}
    pos = [m[v] for v, l in labels["e2_40"].items() if l == "yes" and v in m]
    neg = [m[v] for v, l in labels["e2_40"].items() if l == "no" and v in m]
    row["e2_40"] = {"auc": round(auc(pos, neg), 3), "faithful_mean": round(sum(pos)/len(pos), 4),
                    "unfaithful_mean": round(sum(neg)/len(neg), 4)}
    m = json.load(open(f"{ROOT}/{clip_src['d40'][0]}"))["per_sample"]
    pos = [m[v] for v, l in labels["d40"].items() if l == "yes" and v in m]
    neg = [m[v] for v, l in labels["d40"].items() if l == "no" and v in m]
    row["d40"] = {"auc": round(auc(pos, neg), 3), "faithful_mean": round(sum(pos)/len(pos), 4),
                  "unfaithful_mean": round(sum(neg)/len(neg), 4)}
    out["clip_subject"] = row

    print(f"{'metric':12s} {'宽松AUC':>8} {'约束AUC':>8}")
    for k, v in out.items():
        print(f"{k:12s} {v['e2_40']['auc']:>8.3f} {v['d40']['auc']:>8.3f}")
    json.dump(out, open(f"{ROOT}/demo_2026-07-21/gen_metrics_auc.json", "w"), indent=2)
    print("->", "demo_2026-07-21/gen_metrics_auc.json")


if __name__ == "__main__":
    main()
