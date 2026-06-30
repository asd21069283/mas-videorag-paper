# batch_eval.py
# 批量跑 N 个 V-STaR 样本建 dev 集结果(分阶段: 每个模型只加载一次, 省时)。
# 阶段: A CLIP选帧 -> B Qwen理解+grounding -> C SDXL生成 -> D CLIP评测+gold IoU -> 汇总
# 用法: python pipeline/batch_eval.py --n 50 --out /root/autodl-tmp/dev_eval
import os, json, argparse, traceback
os.environ.setdefault("HF_HOME", "/root/autodl-tmp/hf")
os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")
import sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from vkig_pipeline import read_frames, select_by_score, parse_qwen_bbox

VID_DIR = "/root/autodl-tmp/vstar/videos/videos"
ANN = "/root/autodl-tmp/vstar/V_STaR_test.json"
QWEN = "/root/autodl-tmp/Qwen3-VL-4B-Instruct"
CLIP = "openai/clip-vit-base-patch32"
EDIT = "re-render in a new cinematic scene at night, rain reflections"


def iou2d(a, b):
    x1 = max(a[0], b[0]); y1 = max(a[1], b[1]); x2 = min(a[2], b[2]); y2 = min(a[3], b[3])
    inter = max(0, x2 - x1) * max(0, y2 - y1)
    ua = (a[2]-a[0])*(a[3]-a[1]) + (b[2]-b[0])*(b[3]-b[1]) - inter
    return inter / ua if ua > 0 else 0.0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=50)
    ap.add_argument("--out", default="/root/autodl-tmp/dev_eval")
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    import torch, torch.nn.functional as F
    from PIL import Image
    if torch.cuda.is_available():
        torch.zeros(1, device="cuda")          # 先占CUDA上下文(防PyAV后cuda抽风)

    data = json.load(open(ANN, encoding="utf-8"))
    # 取前 N 个有视频的样本(每个vid一条)
    samples, seen = [], set()
    for s in data:
        v = str(s["vid"])
        if v in seen:
            continue
        p = f"{VID_DIR}/{v}.mp4"
        if os.path.exists(p):
            seen.add(v); samples.append((s, p))
        if len(samples) >= a.n:
            break
    print(f"[batch] {len(samples)} 个样本有视频")
    R = [{"vid": str(s["vid"]), "query": s["question"], "object": s.get("object", ""),
          "video": p, "interval": s.get("timestamps"), "gold_boxes": s.get("bboxes", [])} for s, p in samples]

    # ---- A: CLIP 选帧 ----
    from transformers import CLIPModel, CLIPProcessor
    cm = CLIPModel.from_pretrained(CLIP, use_safetensors=True).to("cuda").eval()
    cp = CLIPProcessor.from_pretrained(CLIP)
    for r in R:
        try:
            frames = read_frames(r["video"], fps=1.0)
            imgs = [f[1] for f in frames]
            inp = cp(text=[r["query"]], images=imgs, return_tensors="pt", padding=True, truncation=True).to("cuda")
            with torch.no_grad():
                o = cm(**inp)
            sc = F.normalize(o.image_embeds, -1) @ F.normalize(o.text_embeds, -1).T
            sc = sc.squeeze(-1).tolist()
            idx = select_by_score(sc, [f[0] for f in frames], k=4)[0]
            r["kf_ts"] = frames[idx][0]; r["_kf"] = frames[idx][1]
        except Exception as e:
            r["err_A"] = str(e)[:120]; r["_kf"] = None
    del cm; import gc; gc.collect(); torch.cuda.empty_cache()
    print("[A] 选帧完成")

    # ---- B: Qwen 理解 + grounding ----
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
            r["answer"] = qask([r["_kf"]], r["query"])
            r["subject"] = qask([r["_kf"]], "Name the single main object the question is about, short phrase (type+color). Output only the phrase.")
            graw = qask([r["_kf"]], f'Locate the {r["object"] or r["subject"]} in this image. Output ONLY JSON {{"bbox_2d":[x1,y1,x2,y2]}} integers 0-1000, top-left origin.')
            r["pred_box"] = parse_qwen_bbox(graw, r["_kf"].width, r["_kf"].height)
        except Exception as e:
            r["err_B"] = str(e)[:120]
    del qm; gc.collect(); torch.cuda.empty_cache()
    print("[B] 理解+定位完成")

    # ---- C: SDXL 生成 ----
    from diffusers import AutoPipelineForImage2Image
    sd = AutoPipelineForImage2Image.from_pretrained("stabilityai/sdxl-turbo", torch_dtype=torch.float16, variant="fp16").to("cuda")
    sd.set_progress_bar_config(disable=True)
    gendir = os.path.join(a.out, "gen"); os.makedirs(gendir, exist_ok=True)
    for r in R:
        if r.get("_kf") is None:
            continue
        try:
            subj = r.get("subject") or r.get("object") or "the main object"
            r["instruction"] = f"{subj}, {EDIT}"
            g = sd(prompt=r["instruction"], image=r["_kf"].convert("RGB"), num_inference_steps=6, strength=0.6, guidance_scale=0.0).images[0]
            gp = os.path.join(gendir, f'{r["vid"]}.png'); g.save(gp); r["_gen"] = g
        except Exception as e:
            r["err_C"] = str(e)[:120]; r["_gen"] = None
    del sd; gc.collect(); torch.cuda.empty_cache()
    print("[C] 生成完成")

    # ---- D: CLIP 评测 + gold IoU ----
    cm = CLIPModel.from_pretrained(CLIP, use_safetensors=True).to("cuda").eval()
    for r in R:
        if r.get("_kf") is None or r.get("_gen") is None:
            continue
        try:
            inp = cp(text=[r["instruction"]], images=[r["_kf"].convert("RGB"), r["_gen"].convert("RGB")],
                     return_tensors="pt", padding=True, truncation=True).to("cuda")
            with torch.no_grad():
                o = cm(**inp)
            ie = F.normalize(o.image_embeds, -1); te = F.normalize(o.text_embeds, -1)
            r["clip_subject_sim"] = round(float((ie[0]*ie[1]).sum()), 4)
            r["clip_text_align"] = round(float((ie[1]*te[0]).sum()), 4)
        except Exception as e:
            r["err_D"] = str(e)[:120]
        # gold IoU + 时间命中
        if r.get("pred_box") and r.get("gold_boxes"):
            near = min(r["gold_boxes"], key=lambda b: abs(float(b["timestamp"]) - r.get("kf_ts", 0)))
            gold = [near["xmin"], near["ymin"], near["xmax"], near["ymax"]]
            r["spatial_iou"] = round(iou2d(r["pred_box"], gold), 3)
        if r.get("interval") and r.get("kf_ts") is not None:
            r["temporal_hit"] = bool(r["interval"][0] <= r["kf_ts"] <= r["interval"][1])
    del cm; gc.collect(); torch.cuda.empty_cache()
    print("[D] 评测完成")

    # ---- 汇总 ----
    def avg(key, cond=lambda r: True):
        vs = [r[key] for r in R if key in r and cond(r)]
        return round(sum(vs)/len(vs), 4) if vs else None
    done = [r for r in R if r.get("_gen") is not None]
    summary = {
        "n_total": len(R), "n_done": len(done),
        "mean_spatial_iou": avg("spatial_iou"),
        "localization_acc@0.5": round(sum(1 for r in R if r.get("spatial_iou", 0) >= 0.5)/len(R), 4) if R else None,
        "localization_acc@0.3": round(sum(1 for r in R if r.get("spatial_iou", 0) >= 0.3)/len(R), 4) if R else None,
        "temporal_recall": round(sum(1 for r in R if r.get("temporal_hit"))/len(R), 4) if R else None,
        "mean_clip_subject_sim": avg("clip_subject_sim"),
        "mean_clip_text_align": avg("clip_text_align"),
    }
    for r in R:
        r.pop("_kf", None); r.pop("_gen", None); r.pop("gold_boxes", None)
    json.dump({"summary": summary, "per_sample": R}, open(os.path.join(a.out, "dev_results.json"), "w", encoding="utf-8"), ensure_ascii=False, indent=2)
    print("=== SUMMARY ==="); print(json.dumps(summary, ensure_ascii=False, indent=2))
    print("[done] ->", os.path.join(a.out, "dev_results.json"))


if __name__ == "__main__":
    main()
