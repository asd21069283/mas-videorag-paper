# batch_eval.py
# 批量跑 N 个 V-STaR 样本建 dev 集结果(分阶段: 每个模型只加载一次, 省时)。
# 阶段: A CLIP选帧 -> B Qwen理解+grounding -> C SDXL生成 -> D CLIP评测+gold IoU -> 汇总
# 用法: python pipeline/batch_eval.py --n 50 --out /root/autodl-tmp/dev_eval
import os, json, argparse, traceback
os.environ.setdefault("HF_HOME", "/root/autodl-tmp/hf")
os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")
import sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from vkig_pipeline import read_frames, select_by_score, parse_qwen_bbox, smooth_scores, combine_scores

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
    ap.add_argument("--ground", default="qwen", choices=["qwen", "gdino"],
                    help="定位用哪个: qwen(自带) / gdino(Grounding-DINO, 型号见 GDINO_ID 环境变量)")
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
            texts = [r["query"], r["object"]] if r.get("object") else [r["query"]]
            inp = cp(text=texts, images=imgs, return_tensors="pt", padding=True, truncation=True).to("cuda")
            with torch.no_grad():
                o = cm(**inp)
            sims = F.normalize(o.image_embeds, dim=-1) @ F.normalize(o.text_embeds, dim=-1).T  # [N,T]
            per = sims.T.tolist()
            comb = combine_scores(per[0], per[1], w_obj=0.6) if len(per) > 1 else per[0]  # 问题+物体联合
            sm = smooth_scores(comb, win=3)                    # 平滑抑尖峰
            idx = max(range(len(sm)), key=lambda i: sm[i])     # 主关键帧=联合分峰值
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
            if a.ground == "qwen":
                graw = qask([r["_kf"]], f'Locate the {r["object"] or r["subject"]} in this image. Output ONLY JSON {{"bbox_2d":[x1,y1,x2,y2]}} integers 0-1000, top-left origin.')
                r["pred_box"] = parse_qwen_bbox(graw, r["_kf"].width, r["_kf"].height)
        except Exception as e:
            r["err_B"] = str(e)[:120]
    del qm; gc.collect(); torch.cuda.empty_cache()
    print("[B] 理解完成 (grounding=%s)" % a.ground)

    # ---- B2: Grounding-DINO 定位(可选, 型号 GDINO_ID; 一次加载跑全部) ----
    if a.ground == "gdino":
        gid = os.environ.get("GDINO_ID", "IDEA-Research/grounding-dino-base")
        from transformers import AutoProcessor as GProc, AutoModelForZeroShotObjectDetection
        gp = GProc.from_pretrained(gid)
        gmodel = AutoModelForZeroShotObjectDetection.from_pretrained(gid).to("cuda").eval()
        for r in R:
            if r.get("_kf") is None:
                continue
            try:
                txt = (r["object"] or r.get("subject") or "object").lower().strip()
                if not txt.endswith("."):
                    txt += "."
                inp = gp(images=r["_kf"], text=txt, return_tensors="pt").to("cuda")
                with torch.no_grad():
                    out = gmodel(**inp)
                res = gp.post_process_grounded_object_detection(out, inp.input_ids, threshold=0.25, text_threshold=0.2, target_sizes=[r["_kf"].size[::-1]])[0]
                if len(res["scores"]):
                    j = int(res["scores"].argmax())
                    r["pred_box"] = [float(v) for v in res["boxes"][j].tolist()]
            except Exception as e:
                r["err_B2"] = str(e)[:120]
        del gmodel; gc.collect(); torch.cuda.empty_cache()
        print(f"[B2] Grounding-DINO({gid}) 定位完成")

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
            ie = F.normalize(o.image_embeds, dim=-1); te = F.normalize(o.text_embeds, dim=-1)
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
    N = len(R) or 1
    # 主定位指标 = 时间命中 AND 空间IoU达标(与 eval/spatiotemporal_iou.py 口径一致); 只看空间的记为 diag(会偏乐观)
    def jointacc(thr): return round(sum(1 for r in R if r.get("temporal_hit") and r.get("spatial_iou", 0) >= thr)/N, 4)
    def lenient(thr): return round(sum(1 for r in R if r.get("spatial_iou", 0) >= thr)/N, 4)
    hit = [r["spatial_iou"] for r in R if r.get("temporal_hit") and "spatial_iou" in r]
    summary = {
        "ground": a.ground + (":" + os.environ.get("GDINO_ID", "grounding-dino-base") if a.ground == "gdino" else ""),
        "n_total": len(R), "n_done": len(done),
        "temporal_recall": round(sum(1 for r in R if r.get("temporal_hit"))/N, 4),
        # === 主指标(诚实: 时间+空间都对) ===
        "localization_acc@0.3_JOINT": jointacc(0.3),
        "localization_acc@0.5_JOINT": jointacc(0.5),
        "spatial_iou_when_temporal_hit": round(sum(hit)/len(hit), 4) if hit else None,
        # === 诊断(只看空间, 拿最近时刻gold框, 偏乐观, 勿当主指标) ===
        "diag_mean_spatial_iou_nearest": avg("spatial_iou"),
        "diag_spatial_acc@0.3_lenient": lenient(0.3),
        "diag_spatial_acc@0.5_lenient": lenient(0.5),
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
