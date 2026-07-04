# hcstvg_eval.py
# HC-STVG v2 定位主实验(§6.4): 读 VKIG jsonl -> 选帧 -> Qwen自带grounding -> 时空联合IoU。
# 只做定位(不生成图, 生成留给 FLUX 专场)。指标口径与 batch_eval / eval/spatiotemporal_iou 完全一致:
#   主指标 = 时间命中 AND 空间IoU>=thr(诚实); 只看空间的记为 diag(偏乐观)。
# 用法: python pipeline/hcstvg_eval.py --vkig /root/autodl-tmp/hcstvg2/vkig_train.jsonl \
#         --video_dir /root/autodl-tmp/hcstvg2/v2_video --n 150 --out /root/autodl-tmp/hcstvg_eval
import os, json, argparse
os.environ.setdefault("HF_HOME", "/root/autodl-tmp/hf")
os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")
import sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from vkig_pipeline import read_frames, smooth_scores, combine_scores, parse_qwen_bbox

QWEN = os.environ.get("QWEN_DIR", "/root/autodl-tmp/Qwen3-VL-4B-Instruct")
CLIP = "openai/clip-vit-base-patch32"
_VIDEXTS = (".mp4", ".mkv", ".webm", ".avi", ".mov")


def iou2d(a, b):
    x1 = max(a[0], b[0]); y1 = max(a[1], b[1]); x2 = min(a[2], b[2]); y2 = min(a[3], b[3])
    inter = max(0, x2 - x1) * max(0, y2 - y1)
    ua = (a[2]-a[0])*(a[3]-a[1]) + (b[2]-b[0])*(b[3]-b[1]) - inter
    return inter / ua if ua > 0 else 0.0


def resolve_video(video_field, video_dir):
    """VKIG 里 video 可能带 .mp4 后缀但真实文件是 .mkv, 按 stem 找。"""
    stem = os.path.splitext(os.path.basename(str(video_field)))[0]
    for ext in _VIDEXTS:
        p = os.path.join(video_dir, stem + ext)
        if os.path.exists(p):
            return p
    return None


def load_vkig(path, video_dir, n):
    """返回 (R, n_missing_video)。n_missing_video = 有gold但视频没下(被跳过)的样本数,
    用于诚实汇报"分母是可用视频样本,不是数据集前N条"(Codex审计第7条)。"""
    R = []; n_missing = 0
    for line in open(path, encoding="utf-8"):
        line = line.strip()
        if not line:
            continue
        s = json.loads(line)
        ev = (s.get("gold", {}) or {}).get("evidence") or []
        if not ev:
            continue
        e = ev[0]
        vp = resolve_video(s.get("video", ""), video_dir)
        if not vp:                                   # 该样本视频没下 -> 计数并跳过(不进分母)
            n_missing += 1
            continue
        track = e.get("track") or []
        gold = [{"timestamp": t["ts"], "xmin": t["bbox"][0], "ymin": t["bbox"][1],
                 "xmax": t["bbox"][2], "ymax": t["bbox"][3]} for t in track if t.get("bbox")]
        R.append({"vid": os.path.splitext(os.path.basename(vp))[0],
                  "query": s.get("query", ""), "object": s.get("object", ""),
                  "video": vp, "interval": e.get("interval"), "gold_boxes": gold})
        if len(R) >= n:
            break
    return R, n_missing


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--vkig", required=True, help="hcstvg_to_vkig 产出的 jsonl")
    ap.add_argument("--video_dir", required=True)
    ap.add_argument("--n", type=int, default=150)
    ap.add_argument("--out", default="/root/autodl-tmp/hcstvg_eval")
    ap.add_argument("--w_obj", type=float, default=0.6, help="物体信号在联合选帧里的权重")
    ap.add_argument("--t_pad", type=float, default=0.0,
                    help="时间命中容差(秒): 0=严格(与V-STaR batch_eval一致,保守); "
                         "eval/spatiotemporal_iou.py默认0.5。诊断值另按0.5额外算一份。")
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    import torch, torch.nn.functional as F
    if torch.cuda.is_available():
        torch.zeros(1, device="cuda")                # 先占CUDA上下文(防PyAV后抽风)

    R, n_missing_video = load_vkig(a.vkig, a.video_dir, a.n)
    print(f"[hcstvg] {len(R)} 个可用样本(视频已就绪); 因缺视频跳过 {n_missing_video} 条")
    # 抽查 object 抽取质量(Codex审计第4条: caption_to_np是启发式)
    _obj_ex = [r["object"] for r in R[:5]]
    _obj_degenerate = sum(1 for r in R if r.get("object") and len(r["object"].split()) > 8)
    print(f"[obj] 示例={_obj_ex} | 疑似退化为长句(>8词)={_obj_degenerate}/{len(R)}")
    if not R:
        print("[abort] 没有可用样本: 检查 --video_dir 下是否有对应视频")
        return

    # ---- A: CLIP 物体条件选帧 ----
    from transformers import CLIPModel, CLIPProcessor
    cm = CLIPModel.from_pretrained(CLIP, use_safetensors=True).to("cuda").eval()
    cp = CLIPProcessor.from_pretrained(CLIP)
    for r in R:
        try:
            frames = read_frames(r["video"], fps=1.0)
            if not frames:
                r["_kf"] = None; r["err_A"] = "no_frames"; continue
            imgs = [f[1] for f in frames]
            texts = [r["query"], r["object"]] if r.get("object") else [r["query"]]
            inp = cp(text=texts, images=imgs, return_tensors="pt", padding=True, truncation=True).to("cuda")
            with torch.no_grad():
                o = cm(**inp)
            sims = F.normalize(o.image_embeds, dim=-1) @ F.normalize(o.text_embeds, dim=-1).T
            per = sims.T.tolist()
            comb = combine_scores(per[0], per[1], w_obj=a.w_obj) if len(per) > 1 else per[0]
            sm = smooth_scores(comb, win=3)
            idx = max(range(len(sm)), key=lambda i: sm[i])
            r["kf_ts"] = frames[idx][0]; r["_kf"] = frames[idx][1]
        except Exception as ex:
            r["_kf"] = None; r["err_A"] = str(ex)[:120]
    del cm; import gc; gc.collect(); torch.cuda.empty_cache()
    print("[A] 选帧完成")

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
            r["pred_box"] = parse_qwen_bbox(graw, r["_kf"].width, r["_kf"].height)   # 默认 norm1000(prompt明确)
        except Exception as ex:
            r["err_B"] = str(ex)[:120]
    del qm; gc.collect(); torch.cuda.empty_cache()
    print("[B] 定位完成")

    # ---- 联合 IoU ----
    # 空间gold口径: 从gold逐帧tube取"预测关键帧时刻最近的框"(Codex审计第2条: 对HC-STVG轨迹比固定中点框更合理;
    #   JOINT下要求temporal_hit=kf_ts落在区间内, 此时最近gold框≈预测时刻真值, 公平)。
    def temporal_ok(r, pad):
        return bool(r["interval"][0] - pad <= r["kf_ts"] <= r["interval"][1] + pad)
    for r in R:
        if r.get("pred_box") and r.get("gold_boxes"):
            near = min(r["gold_boxes"], key=lambda b: abs(float(b["timestamp"]) - r.get("kf_ts", 0)))
            gold = [near["xmin"], near["ymin"], near["xmax"], near["ymax"]]
            r["spatial_iou"] = round(iou2d(r["pred_box"], gold), 3)
        if r.get("interval") and r.get("kf_ts") is not None:
            r["temporal_hit"] = temporal_ok(r, a.t_pad)             # 主口径(默认严格t_pad=0)
            r["temporal_hit_pad05"] = temporal_ok(r, 0.5)           # 诊断: ±0.5s容差(与score_sample默认一致)

    # ---- 汇总(与 batch_eval 同口径; 缺视频/pad 已诚实标注) ----
    N = len(R) or 1
    def jointacc(thr, pad_key="temporal_hit"):
        return round(sum(1 for r in R if r.get(pad_key) and r.get("spatial_iou", 0) >= thr)/N, 4)
    def lenient(thr): return round(sum(1 for r in R if r.get("spatial_iou", 0) >= thr)/N, 4)
    hit = [r["spatial_iou"] for r in R if r.get("temporal_hit") and "spatial_iou" in r]
    def avg(key):
        vs = [r[key] for r in R if key in r]
        return round(sum(vs)/len(vs), 4) if vs else None
    summary = {
        "dataset": "HC-STVG-v2-test", "ground": "qwen", "w_obj": a.w_obj,
        "temporal_pad_sec": a.t_pad,                                # 主口径容差(诚实标注)
        "spatial_gold": "tube-box-nearest-to-predicted-keyframe",  # 空间gold口径(非score_sample固定中点)
        "n_video_ready": len(R), "n_missing_video": n_missing_video,  # 分母=可用视频样本, 非数据集前N条
        "n_no_keyframe": sum(1 for r in R if r.get("_kf") is None),
        "temporal_recall": round(sum(1 for r in R if r.get("temporal_hit"))/N, 4),
        "localization_acc@0.3_JOINT": jointacc(0.3),
        "localization_acc@0.5_JOINT": jointacc(0.5),
        "spatial_iou_when_temporal_hit": round(sum(hit)/len(hit), 4) if hit else None,
        # 诊断: ±0.5s容差下的联合指标(看时间口径敏感度)
        "diag_temporal_recall_pad05": round(sum(1 for r in R if r.get("temporal_hit_pad05"))/N, 4),
        "diag_localization_acc@0.3_JOINT_pad05": jointacc(0.3, "temporal_hit_pad05"),
        # 诊断: 只看空间(偏乐观, 勿当主指标)
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
