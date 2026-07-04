# hcstvg_baseline.py
# HC-STVG v2 定位下界基线(§6.5): 读 VKIG jsonl -> 平凡/Oracle选帧 -> Qwen自带grounding -> 时空联合IoU。
# 与 pipeline/hcstvg_eval.py 同口径统计, 唯一区别是 A 步选帧策略不使用 CLIP 物体条件信号。
#
# 用法:
#   python pipeline/hcstvg_baseline.py --vkig /root/autodl-tmp/hcstvg2/vkig_train.jsonl \
#     --video_dir /root/autodl-tmp/hcstvg2/v2_video --n 150 \
#     --sel uniform --out /root/autodl-tmp/hcstvg_baseline_uniform
#   python pipeline/hcstvg_baseline.py --vkig /root/autodl-tmp/hcstvg2/vkig_train.jsonl \
#     --video_dir /root/autodl-tmp/hcstvg2/v2_video --n 150 \
#     --sel random --out /root/autodl-tmp/hcstvg_baseline_random
#   python pipeline/hcstvg_baseline.py --vkig /root/autodl-tmp/hcstvg2/vkig_train.jsonl \
#     --video_dir /root/autodl-tmp/hcstvg2/v2_video --n 150 \
#     --sel middle_gt --out /root/autodl-tmp/hcstvg_baseline_middle_gt
import os, json, argparse, random

os.environ.setdefault("HF_HOME", "/root/autodl-tmp/hf")
os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")

import sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from hcstvg_eval import QWEN, iou2d, load_vkig
from vkig_pipeline import read_frames, parse_qwen_bbox


def select_keyframe(frames, strategy, interval, rng):
    """从 read_frames 的候选帧里选 1 帧; 不看图像内容。返回 (ts, image)。"""
    if not frames:
        return None
    if strategy == "random":
        return rng.choice(frames)

    if strategy == "middle_gt":
        if not interval or len(interval) < 2:
            return None
        target = (float(interval[0]) + float(interval[1])) / 2.0
    else:
        target = (float(frames[0][0]) + float(frames[-1][0])) / 2.0

    return min(frames, key=lambda f: abs(float(f[0]) - target))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--vkig", required=True, help="hcstvg_to_vkig 产出的 jsonl")
    ap.add_argument("--video_dir", required=True)
    ap.add_argument("--n", type=int, default=150)
    ap.add_argument("--out", default="/root/autodl-tmp/hcstvg_baseline")
    ap.add_argument("--sel", choices=["uniform", "random", "middle_gt"], default="uniform",
                    help="选帧策略: uniform=视频时间中点; random=固定种子随机帧; middle_gt=gold区间中点oracle")
    ap.add_argument("--t_pad", type=float, default=0.0,
                    help="时间命中容差(秒): 0=严格(与V-STaR batch_eval一致,保守); "
                         "eval/spatiotemporal_iou.py默认0.5。诊断值另按0.5额外算一份。")
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)

    import torch
    if torch.cuda.is_available():
        torch.zeros(1, device="cuda")

    R, n_missing_video = load_vkig(a.vkig, a.video_dir, a.n)
    print(f"[hcstvg-baseline:{a.sel}] {len(R)} 个可用样本(视频已就绪); 因缺视频跳过 {n_missing_video} 条")
    _obj_ex = [r["object"] for r in R[:5]]
    _obj_degenerate = sum(1 for r in R if r.get("object") and len(r["object"].split()) > 8)
    print(f"[obj] 示例={_obj_ex} | 疑似退化为长句(>8词)={_obj_degenerate}/{len(R)}")
    if not R:
        print("[abort] 没有可用样本: 检查 --video_dir 下是否有对应视频")
        return

    # ---- A: 平凡/Oracle 选帧 ----
    rng = random.Random(42)
    for r in R:
        try:
            frames = read_frames(r["video"], fps=1.0)
            if not frames:
                r["_kf"] = None; r["err_A"] = "no_frames"; continue
            picked = select_keyframe(frames, a.sel, r.get("interval"), rng)
            if picked is None:
                r["_kf"] = None; r["err_A"] = "no_selectable_frame"; continue
            r["kf_ts"] = picked[0]; r["_kf"] = picked[1]
        except Exception as ex:
            r["_kf"] = None; r["err_A"] = str(ex)[:120]
    print(f"[A] {a.sel} 选帧完成")

    # ---- B: Qwen 自带 grounding ----
    from transformers import AutoProcessor
    try:
        from transformers import AutoModelForImageTextToText as VLM
    except Exception:
        from transformers import Qwen3VLForConditionalGeneration as VLM
    from qwen_vl_utils import process_vision_info
    proc = AutoProcessor.from_pretrained(QWEN)
    qm = VLM.from_pretrained(QWEN, torch_dtype="auto", device_map="cuda").eval()

    def qask(imgs, q):
        msgs = [{"role": "user", "content": [{"type": "image", "image": im} for im in imgs] + [{"type": "text", "text": q}]}]
        text = proc.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
        ii, vi = process_vision_info(msgs)
        inp = proc(text=[text], images=ii, videos=vi, padding=True, return_tensors="pt").to("cuda")
        with torch.no_grad():
            g = qm.generate(**inp, max_new_tokens=128, do_sample=False)
        return proc.batch_decode(g[:, inp.input_ids.shape[1]:], skip_special_tokens=True)[0].strip()

    for r in R:
        if r.get("_kf") is None:
            continue
        try:
            obj = r["object"] or "the main person"
            graw = qask([r["_kf"]], f'Locate the {obj} in this image. Output ONLY JSON '
                                    f'{{"bbox_2d":[x1,y1,x2,y2]}} integers 0-1000, top-left origin.')
            r["pred_box"] = parse_qwen_bbox(graw, r["_kf"].width, r["_kf"].height)
        except Exception as ex:
            r["err_B"] = str(ex)[:120]
    del qm; import gc; gc.collect(); torch.cuda.empty_cache()
    print("[B] 定位完成")

    # ---- 联合 IoU ----
    def temporal_ok(r, pad):
        return bool(r["interval"][0] - pad <= r["kf_ts"] <= r["interval"][1] + pad)
    for r in R:
        if r.get("pred_box") and r.get("gold_boxes"):
            near = min(r["gold_boxes"], key=lambda b: abs(float(b["timestamp"]) - r.get("kf_ts", 0)))
            gold = [near["xmin"], near["ymin"], near["xmax"], near["ymax"]]
            r["spatial_iou"] = round(iou2d(r["pred_box"], gold), 3)
        if r.get("interval") and r.get("kf_ts") is not None:
            r["temporal_hit"] = temporal_ok(r, a.t_pad)
            r["temporal_hit_pad05"] = temporal_ok(r, 0.5)

    # ---- 汇总(字段名与 hcstvg_eval.py 保持一致, 仅新增 sel_strategy) ----
    N = len(R) or 1
    def jointacc(thr, pad_key="temporal_hit"):
        return round(sum(1 for r in R if r.get(pad_key) and r.get("spatial_iou", 0) >= thr)/N, 4)
    def lenient(thr): return round(sum(1 for r in R if r.get("spatial_iou", 0) >= thr)/N, 4)
    hit = [r["spatial_iou"] for r in R if r.get("temporal_hit") and "spatial_iou" in r]
    def avg(key):
        vs = [r[key] for r in R if key in r]
        return round(sum(vs)/len(vs), 4) if vs else None
    summary = {
        "dataset": "HC-STVG-v2-test", "ground": "qwen", "w_obj": None,
        "sel_strategy": a.sel,
        "temporal_pad_sec": a.t_pad,
        "spatial_gold": "tube-box-nearest-to-predicted-keyframe",
        "n_video_ready": len(R), "n_missing_video": n_missing_video,
        "n_no_keyframe": sum(1 for r in R if r.get("_kf") is None),
        "temporal_recall": round(sum(1 for r in R if r.get("temporal_hit"))/N, 4),
        "localization_acc@0.3_JOINT": jointacc(0.3),
        "localization_acc@0.5_JOINT": jointacc(0.5),
        "spatial_iou_when_temporal_hit": round(sum(hit)/len(hit), 4) if hit else None,
        "diag_temporal_recall_pad05": round(sum(1 for r in R if r.get("temporal_hit_pad05"))/N, 4),
        "diag_localization_acc@0.3_JOINT_pad05": jointacc(0.3, "temporal_hit_pad05"),
        "diag_mean_spatial_iou_nearest": avg("spatial_iou"),
        "diag_spatial_acc@0.3_lenient": lenient(0.3),
        "diag_spatial_acc@0.5_lenient": lenient(0.5),
    }
    for r in R:
        r.pop("_kf", None); r.pop("gold_boxes", None)
    json.dump({"summary": summary, "per_sample": R},
              open(os.path.join(a.out, "hcstvg_results.json"), "w", encoding="utf-8"),
              ensure_ascii=False, indent=2)
    print("=== SUMMARY ==="); print(json.dumps(summary, ensure_ascii=False, indent=2))
    print("[done] ->", os.path.join(a.out, "hcstvg_results.json"))


if __name__ == "__main__":
    main()
