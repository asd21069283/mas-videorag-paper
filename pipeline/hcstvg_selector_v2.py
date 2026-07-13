# hcstvg_selector_v2.py — 时序选择器改进实验(主线创新点)
# 对照基线=物体条件选帧(时间召回~0.55)。两个增量假设:
#   E1 三路打分: query+object之外加"动作短语"CLIP通道(动作未被建模是缺口)
#   E2 两阶段: Qwen3-VL看抽稀帧网格粗定位动作时段 -> 段内CLIP物体条件精选(MLLM时序理解>CLIP)
# 只算定位(与hcstvg_eval同口径), 出: 时间召回/联合acc/IoU@hit, 与E0基线可直比。
# 用法: python pipeline/hcstvg_selector_v2.py --vkig ... --video_dir ... --n 300 --exp e1|e2 --out ...
import os, json, argparse, re
os.environ.setdefault("HF_HOME", "/root/autodl-tmp/hf")
os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")
import sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from hcstvg_eval import load_vkig, QWEN, CLIP, iou2d
from vkig_pipeline import read_frames, smooth_scores, combine_scores, parse_qwen_bbox

_VERB = re.compile(
    r"\b(is|are|was|were|turns?|walks?|runs?|sits?|stands?|holds?|moves?|looks?|wears?|raises?|"
    r"puts?|takes?|picks?|points?|waves?|bends?|jumps?|lies?|talks?|opens?|closes?|"
    r"lying|sitting|standing|walking|holding|wearing|turning|moving|looking)\b", re.I)


def action_phrase(caption, obj):
    """动作短语 = 第一个动词起的剩余句(如 'turns around and walks to the door')。抽不到退回整句。"""
    cap = str(caption).strip().rstrip(".")
    m = _VERB.search(cap)
    act = cap[m.start():].strip() if m else cap
    return f"a person {act}" if act and not act.lower().startswith(("a ", "the ")) else act or cap


def pick_peak(scores, ts_list, win=3):
    sm = smooth_scores(scores, win=win)
    i = max(range(len(sm)), key=lambda k: sm[k])
    return ts_list[i], i


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--vkig", required=True); ap.add_argument("--video_dir", required=True)
    ap.add_argument("--n", type=int, default=300); ap.add_argument("--out", required=True)
    ap.add_argument("--exp", choices=["e1", "e2"], required=True)
    ap.add_argument("--w_obj", type=float, default=0.4, help="e1: 物体通道权重")
    ap.add_argument("--w_act", type=float, default=0.3, help="e1: 动作通道权重(问题=1-w_obj-w_act)")
    ap.add_argument("--grid_n", type=int, default=10, help="e2: 给Qwen看的抽稀帧数")
    ap.add_argument("--t_pad", type=float, default=0.0)
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    import torch, torch.nn.functional as F
    if torch.cuda.is_available():
        torch.zeros(1, device="cuda")

    R, n_missing = load_vkig(a.vkig, a.video_dir, a.n)
    print(f"[{a.exp}] {len(R)} 可用样本, 缺视频 {n_missing}", flush=True)

    # ============ 抽帧(共用) ============
    for r in R:
        try:
            fr = read_frames(r["video"], fps=1.0)
            r["_frames"] = fr if fr else None
        except Exception as e:
            r["_frames"] = None; r["err_read"] = str(e)[:100]

    # ============ E2 第一阶段: Qwen 粗定位时段 ============
    if a.exp == "e2":
        from transformers import AutoProcessor
        try:
            from transformers import AutoModelForImageTextToText as VLM
        except Exception:
            from transformers import Qwen3VLForConditionalGeneration as VLM
        from qwen_vl_utils import process_vision_info
        proc = AutoProcessor.from_pretrained(QWEN)
        qm = VLM.from_pretrained(QWEN, torch_dtype="auto", device_map="cuda").eval()
        for r in R:
            if not r.get("_frames"):
                continue
            fr = r["_frames"]
            # 均匀抽 grid_n 帧, 标号给 Qwen 问动作时段
            idxs = [round(i*(len(fr)-1)/(a.grid_n-1)) for i in range(min(a.grid_n, len(fr)))]
            idxs = sorted(set(idxs))
            imgs = [fr[i][1] for i in idxs]
            q = (f'These {len(imgs)} frames are uniformly sampled from a video in time order (frame 1..{len(imgs)}). '
                 f'In which frames is this happening: "{r["query"]}"? '
                 f'Answer ONLY JSON {{"start_frame": s, "end_frame": e}} with 1-based frame numbers.')
            try:
                msgs = [{"role": "user", "content": [{"type": "image", "image": im} for im in imgs] + [{"type": "text", "text": q}]}]
                text = proc.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
                ii, vi = process_vision_info(msgs)
                inp = proc(text=[text], images=ii, videos=vi, padding=True, return_tensors="pt").to("cuda")
                with torch.no_grad():
                    g = qm.generate(**inp, max_new_tokens=48, do_sample=False)
                raw = proc.batch_decode(g[:, inp.input_ids.shape[1]:], skip_special_tokens=True)[0]
                m = re.search(r'"start_frame"\s*:\s*(\d+).*?"end_frame"\s*:\s*(\d+)', raw, re.S)
                if m:
                    s_i, e_i = int(m.group(1))-1, int(m.group(2))-1
                    s_i = max(0, min(s_i, len(idxs)-1)); e_i = max(s_i, min(e_i, len(idxs)-1))
                    # 映射回原帧索引区间(段边界向外扩半格, 防边界截断)
                    lo = idxs[s_i] - (idxs[1]-idxs[0])//2 if s_i > 0 else 0
                    hi = idxs[e_i] + (idxs[1]-idxs[0])//2 if e_i < len(idxs)-1 else len(fr)-1
                    r["_win"] = (max(0, lo), min(len(fr)-1, hi))
            except Exception as e:
                r["err_stage1"] = str(e)[:100]
        del qm; import gc; gc.collect(); torch.cuda.empty_cache()
        print("[e2] Qwen粗定位完成", flush=True)

    # ============ CLIP 打分选帧 ============
    from transformers import CLIPModel, CLIPProcessor
    cm = CLIPModel.from_pretrained(CLIP, use_safetensors=True).to("cuda").eval()
    cp = CLIPProcessor.from_pretrained(CLIP)
    for r in R:
        if not r.get("_frames"):
            r["_kf"] = None; continue
        fr = r["_frames"]
        lo, hi = r.get("_win", (0, len(fr)-1))
        sub = fr[lo:hi+1] if hi > lo else fr
        imgs = [f[1] for f in sub]
        texts = [r["query"], r["object"] or r["query"]]
        if a.exp == "e1":
            texts.append(action_phrase(r["query"], r["object"]))
        try:
            inp = cp(text=texts, images=imgs, return_tensors="pt", padding=True, truncation=True).to("cuda")
            with torch.no_grad():
                o = cm(**inp)
            sims = F.normalize(o.image_embeds, dim=-1) @ F.normalize(o.text_embeds, dim=-1).T
            per = sims.T.tolist()
            if a.exp == "e1" and len(per) == 3:
                from vkig_pipeline import _znorm
                q_, o_, ac_ = _znorm(per[0]), _znorm(per[1]), _znorm(per[2])
                w_q = 1.0 - a.w_obj - a.w_act
                comb = [w_q*x + a.w_obj*y + a.w_act*z for x, y, z in zip(q_, o_, ac_)]
            else:
                comb = combine_scores(per[0], per[1], w_obj=0.6)
            ts, _ = pick_peak(comb, [f[0] for f in sub])
            r["kf_ts"] = ts
            r["_kf"] = sub[[f[0] for f in sub].index(ts)][1]
        except Exception as e:
            r["_kf"] = None; r["err_clip"] = str(e)[:100]
    del cm; import gc; gc.collect(); torch.cuda.empty_cache()
    print("[clip] 选帧完成", flush=True)

    # ============ Qwen grounding(与基线一致) ============
    from transformers import AutoProcessor
    try:
        from transformers import AutoModelForImageTextToText as VLM
    except Exception:
        from transformers import Qwen3VLForConditionalGeneration as VLM
    from qwen_vl_utils import process_vision_info
    proc = AutoProcessor.from_pretrained(QWEN)
    qm = VLM.from_pretrained(QWEN, torch_dtype="auto", device_map="cuda").eval()
    for r in R:
        if r.get("_kf") is None:
            continue
        try:
            obj = r["object"] or "the main person"
            msgs = [{"role": "user", "content": [{"type": "image", "image": r["_kf"]},
                    {"type": "text", "text": f'Locate the {obj} in this image. Output ONLY JSON {{"bbox_2d":[x1,y1,x2,y2]}} integers 0-1000, top-left origin.'}]}]
            text = proc.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
            ii, vi = process_vision_info(msgs)
            inp = proc(text=[text], images=ii, videos=vi, padding=True, return_tensors="pt").to("cuda")
            import torch as _t
            with _t.no_grad():
                g = qm.generate(**inp, max_new_tokens=64, do_sample=False)
            raw = proc.batch_decode(g[:, inp.input_ids.shape[1]:], skip_special_tokens=True)[0]
            r["pred_box"] = parse_qwen_bbox(raw, r["_kf"].width, r["_kf"].height)
        except Exception as e:
            r["err_B"] = str(e)[:100]
    del qm; import gc; gc.collect(); torch.cuda.empty_cache()
    print("[B] 定位完成", flush=True)

    # ============ 联合评测(与hcstvg_eval同口径) ============
    for r in R:
        if r.get("pred_box") and r.get("gold_boxes"):
            near = min(r["gold_boxes"], key=lambda b: abs(float(b["timestamp"]) - r.get("kf_ts", 0)))
            r["spatial_iou"] = round(iou2d(r["pred_box"], [near["xmin"], near["ymin"], near["xmax"], near["ymax"]]), 3)
        if r.get("interval") and r.get("kf_ts") is not None:
            r["temporal_hit"] = bool(r["interval"][0] - a.t_pad <= r["kf_ts"] <= r["interval"][1] + a.t_pad)
    N = len(R) or 1
    def ja(t): return round(sum(1 for r in R if r.get("temporal_hit") and r.get("spatial_iou", 0) >= t)/N, 4)
    hit = [r.get("spatial_iou", 0.0) for r in R if r.get("temporal_hit")]
    summary = {
        "exp": a.exp, "params": {"w_obj": a.w_obj, "w_act": a.w_act, "grid_n": a.grid_n},
        "n": len(R),
        "temporal_recall": round(sum(1 for r in R if r.get("temporal_hit"))/N, 4),
        "localization_acc@0.3_JOINT": ja(0.3), "localization_acc@0.5_JOINT": ja(0.5),
        "spatial_iou_when_temporal_hit": round(sum(hit)/len(hit), 4) if hit else None,
        "n_no_keyframe": sum(1 for r in R if r.get("_kf") is None),
        "n_stage1_window": sum(1 for r in R if "_win" in r),
    }
    for r in R:
        r.pop("_frames", None); r.pop("_kf", None); r.pop("gold_boxes", None)
    json.dump({"summary": summary, "per_sample": R}, open(os.path.join(a.out, "results.json"), "w", encoding="utf-8"), ensure_ascii=False, indent=2)
    print("=== SUMMARY ===");  print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
