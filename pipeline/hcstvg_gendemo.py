# hcstvg_gendemo.py
# 生成demo: HC-STVG 定位到的关键帧 -> FLUX-Kontext(本地目录) / SDXL-Turbo 生成新图 + CLIP指标。
# 补齐"定位→生成"这条链的图像证据(论文§6.4生成部分)。默认优先本地 FLUX, 失败回退 SDXL。
# 用法:
#   FLUX_DIR=/root/autodl-tmp/flux_kontext python pipeline/hcstvg_gendemo.py \
#     --vkig /root/autodl-tmp/hcstvg2/vkig_test.jsonl --video_dir /root/autodl-tmp/hcstvg2/v2_video \
#     --n 8 --out /root/autodl-tmp/hcstvg_gendemo [--sdxl]
import os, json, argparse
os.environ.setdefault("HF_HOME", "/root/autodl-tmp/hf")
os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")
import sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from hcstvg_eval import load_vkig, QWEN, CLIP
from vkig_pipeline import read_frames, smooth_scores, combine_scores, parse_qwen_bbox

FLUX_DIR = os.environ.get("FLUX_DIR", "/root/autodl-tmp/flux_kontext")
EDIT = os.environ.get("VKIG_EDIT", "re-render in a new cinematic scene at night, rain reflections")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--vkig", required=True)
    ap.add_argument("--video_dir", required=True)
    ap.add_argument("--n", type=int, default=8)
    ap.add_argument("--out", default="/root/autodl-tmp/hcstvg_gendemo")
    ap.add_argument("--w_obj", type=float, default=0.6)
    ap.add_argument("--sdxl", action="store_true", help="强制用SDXL(不试FLUX)")
    ap.add_argument("--steps", type=int, default=28)
    a = ap.parse_args()
    gendir = os.path.join(a.out, "img"); os.makedirs(gendir, exist_ok=True)
    import torch, torch.nn.functional as F
    from PIL import Image
    if torch.cuda.is_available():
        torch.zeros(1, device="cuda")

    R, n_missing = load_vkig(a.vkig, a.video_dir, a.n)
    print(f"[gendemo] {len(R)} 样本; 缺视频跳过 {n_missing}")
    if not R:
        print("[abort] 无可用样本"); return

    # ---- A: 物体条件选帧(与主实验一致) ----
    from transformers import CLIPModel, CLIPProcessor
    cm = CLIPModel.from_pretrained(CLIP, use_safetensors=True).to("cuda").eval()
    cp = CLIPProcessor.from_pretrained(CLIP)
    for r in R:
        try:
            frames = read_frames(r["video"], fps=1.0)
            if not frames:
                r["_kf"] = None; continue
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
            r["_kf"].convert("RGB").save(os.path.join(gendir, f'{r["vid"]}_keyframe.png'))
        except Exception as e:
            r["_kf"] = None; r["err_A"] = str(e)[:120]
    del cm; import gc; gc.collect(); torch.cuda.empty_cache()
    print("[A] 选帧完成")

    # ---- B: Qwen 主体名(生成指令用) ----
    from transformers import AutoProcessor
    try:
        from transformers import AutoModelForImageTextToText as VLM
    except Exception:
        from transformers import Qwen3VLForConditionalGeneration as VLM
    from qwen_vl_utils import process_vision_info
    proc = AutoProcessor.from_pretrained(QWEN)
    qm = VLM.from_pretrained(QWEN, torch_dtype="auto", device_map="cuda").eval()
    def qask(im, q):
        msgs = [{"role": "user", "content": [{"type": "image", "image": im}, {"type": "text", "text": q}]}]
        text = proc.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
        ii, vi = process_vision_info(msgs)
        inp = proc(text=[text], images=ii, videos=vi, padding=True, return_tensors="pt").to("cuda")
        with torch.no_grad():
            g = qm.generate(**inp, max_new_tokens=48, do_sample=False)
        return proc.batch_decode(g[:, inp.input_ids.shape[1]:], skip_special_tokens=True)[0].strip()
    for r in R:
        if r.get("_kf") is None:
            continue
        try:
            r["subject"] = qask(r["_kf"], "Name the single main person/object, short phrase (type+color). Output only the phrase.")
        except Exception as e:
            r["subject"] = r.get("object") or "the main subject"; r["err_B"] = str(e)[:120]
    del qm; gc.collect(); torch.cuda.empty_cache()
    print("[B] 主体命名完成")

    # ---- C: 生成(优先本地FLUX, 回退SDXL) ----
    pipe = None; gen_model = None
    if not a.sdxl and os.path.isdir(FLUX_DIR) and os.path.exists(os.path.join(FLUX_DIR, "model_index.json")):
        try:
            from diffusers import FluxKontextPipeline
            pipe = FluxKontextPipeline.from_pretrained(FLUX_DIR, torch_dtype=torch.bfloat16)
            pipe.enable_sequential_cpu_offload()      # 24G卡必须逐层offload(model级不够, 会OOM); 慢但能跑
            try:
                pipe.vae.enable_slicing(); pipe.vae.enable_tiling()
            except Exception:
                pass
            gen_model = "flux-kontext"
            print("[C] 用 FLUX-Kontext(本地)")
        except Exception as e:
            print("[C] FLUX加载失败, 回退SDXL:", str(e)[:160]); pipe = None
    if pipe is None:
        from diffusers import AutoPipelineForImage2Image
        pipe = AutoPipelineForImage2Image.from_pretrained("stabilityai/sdxl-turbo", torch_dtype=torch.float16, variant="fp16").to("cuda")
        pipe.set_progress_bar_config(disable=True); gen_model = "sdxl-turbo"
        print("[C] 用 SDXL-Turbo")

    for r in R:
        if r.get("_kf") is None:
            continue
        subj = r.get("subject") or r.get("object") or "the main subject"
        instr = f"{subj}, {EDIT}"; r["instruction"] = instr
        try:
            kf = r["_kf"].convert("RGB")
            if gen_model == "flux-kontext":
                img = pipe(image=kf, prompt=instr, guidance_scale=2.5, num_inference_steps=a.steps).images[0]
            else:
                img = pipe(prompt=instr, image=kf, num_inference_steps=6, strength=0.6, guidance_scale=0.0).images[0]
            img.save(os.path.join(gendir, f'{r["vid"]}_generated.png')); r["_gen"] = img
        except Exception as e:
            r["err_C"] = str(e)[:160]; r["_gen"] = None
    del pipe; gc.collect(); torch.cuda.empty_cache()
    print("[C] 生成完成:", gen_model)

    # ---- D: CLIP 主体一致 + 图文对齐 ----
    cm = CLIPModel.from_pretrained(CLIP, use_safetensors=True).to("cuda").eval()
    subj_sim, text_al = [], []
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
            subj_sim.append(r["clip_subject_sim"]); text_al.append(r["clip_text_align"])
        except Exception as e:
            r["err_D"] = str(e)[:120]
    del cm; gc.collect(); torch.cuda.empty_cache()

    summary = {
        "dataset": "HC-STVG-v2-test", "gen_model": gen_model, "n": len(R),
        "n_generated": sum(1 for r in R if r.get("_gen") is not None),
        "mean_clip_subject_sim": round(sum(subj_sim)/len(subj_sim), 4) if subj_sim else None,
        "mean_clip_text_align": round(sum(text_al)/len(text_al), 4) if text_al else None,
        "edit_prompt": EDIT,
    }
    for r in R:
        r.pop("_kf", None); r.pop("_gen", None); r.pop("gold_boxes", None)
    json.dump({"summary": summary, "per_sample": R},
              open(os.path.join(a.out, "gendemo_results.json"), "w", encoding="utf-8"), ensure_ascii=False, indent=2)
    print("=== SUMMARY ==="); print(json.dumps(summary, ensure_ascii=False, indent=2))
    print("[done] 图片在", gendir)


if __name__ == "__main__":
    main()
