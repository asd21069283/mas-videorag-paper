# analysis_bottleneck.py — §4.5 Analysis 的实证支撑(纯标准库, 0 GPU)
#   命题1: A = R_t · S_t 恒等式 + S_t 跨选择器近乎不变
#   命题2: 位置盲召回 ≈ E[|W|/L]; 外观argmax不超越中心先验; E2增益近乎窗宽无关
import json, os, statistics as st

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
FILES = {
    "random": "demo_2026-07-04/results_json/hcstvg_bl_random.json",
    "uniform(mid)": "demo_2026-07-04/results_json/hcstvg_bl_uniform.json",
    "oracle": "demo_2026-07-04/results_json/hcstvg_bl_middle_gt.json",
    "E0@1000": "demo_2026-07-04/results_json/hcstvg_main1000.json",
    "E1@300": "demo_2026-07-13/sel_e1_300.json",
    "E2@300": "demo_2026-07-13/sel_e2_300.json",
    "E2@1000": "demo_2026-07-13/sel_e2_1000.json",
    "g15": "demo_2026-07-18/sel_e2_g15_fix.json",
    "g20": "demo_2026-07-18/sel_e2_g20_fix.json",
}
L = 20.0


def load(f): return json.load(open(os.path.join(ROOT, f)))["per_sample"]


def main():
    print("=== 命题1: A(τ)=R_t·S_t 恒等式 + S_t 不变性 (τ=0.3) ===")
    for name, f in FILES.items():
        R = load(f); n = len(R)
        A = sum(1 for r in R if r.get("temporal_hit") and r.get("spatial_iou", 0) >= 0.3) / n
        Rt = sum(1 for r in R if r.get("temporal_hit")) / n
        hits = [r for r in R if r.get("temporal_hit")]
        St = sum(1 for r in hits if r.get("spatial_iou", 0) >= 0.3) / len(hits) if hits else 0
        nobox = sum(1 for r in R if r.get("spatial_iou") is None or "spatial_iou" not in r) / n
        ok = "OK" if abs(A - Rt * St) < 1e-9 else "MISMATCH"
        print(f"  {name:14s} A={A:.4f} R_t={Rt:.4f} S_t={St:.4f} R_t*S_t={Rt*St:.4f} [{ok}] no-box={nobox:.1%}")

    print("\n=== 命题2: 位置盲召回 ≈ E[|W|/L]; 外观argmax ≤ 中心先验 ===")
    E0 = load(FILES["E0@1000"])
    widths = [r["interval"][1] - r["interval"][0] for r in E0 if r.get("interval")]
    centers = [(r["interval"][0] + r["interval"][1]) / 2 for r in E0 if r.get("interval")]
    print(f"  gold窗宽 mean={st.mean(widths):.3f}s → E[|W|/L]={st.mean(widths)/L:.4f} (=random期望召回)")
    print(f"  中心 mean/L={st.mean(centers)/L:.4f}")
    for k in ["random", "uniform(mid)", "E0@1000"]:
        R = load(FILES[k]); print(f"  {k:14s} 召回={sum(1 for r in R if r.get('temporal_hit'))/len(R):.4f}")

    print("\n=== 命题2支撑: 按gold窗宽三分位 E0 vs E2 召回增益 (n=1000同clip) ===")
    E2 = load(FILES["E2@1000"])
    pairs = sorted(zip(E0, E2), key=lambda p: p[0]["interval"][1] - p[0]["interval"][0])
    tri = len(pairs) // 3
    for lab, seg in [("narrow", pairs[:tri]), ("medium", pairs[tri:2 * tri]), ("wide", pairs[2 * tri:])]:
        ww = st.mean([(a["interval"][1] - a["interval"][0]) / L for a, b in seg])
        e0r = sum(1 for a, b in seg if a.get("temporal_hit")) / len(seg)
        e2r = sum(1 for a, b in seg if b.get("temporal_hit")) / len(seg)
        print(f"  {lab:7s} |W|/L={ww:.3f}: E0={e0r:.3f} E2={e2r:.3f} gain=+{e2r-e0r:.3f}")


if __name__ == "__main__":
    main()
