# gen_from_e2.py — 用 E2 两阶段选择器已选出的关键帧(sel_e2_300结果)批量跑 FLUX 生成。
# 产物: img/{vid}_keyframe.png + img/{vid}_generated.png + results.json(CLIP两分)
# 供: 生成侧指标(VQAScore后打)、忠实性人评/MLLM评、论文§6生成表。
# 用法: FLUX_DIR=/root/autodl-tmp/flux_kontext python pipeline/gen_from_e2.py \
#   --e2 /root/autodl-tmp/sel_e2_300/results.json --video_dir /root/autodl-tmp/hcstvg2/v2_video \
#   --n 40 --out /root/autodl-tmp/gen_e2_40
import os, json, argparse
os.environ.setdefault("HF_HOME", "/root/autodl-tmp/hf")
os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")
import sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from hcstvg_eval import CLIP
from vkig_pipeline import read_frames

FLUX_DIR = os.environ.get("FLUX_DIR", "/root/autodl-tmp/flux_kontext")
EDIT = os.environ.get("VKIG_EDIT", "re-render as a cinematic promotional poster shot, dramatic lighting")
_VIDEXTS = (".mp4", ".mkv", ".webm", ".avi", ".mov")


def find_video(stem, vdir):
    for ext in _VIDEXTS:
        p = os.path.join(vdir, stem + ext)
        if os.path.exists(p):
            return p
    return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--e2", required=True, help="sel_e2_*/results.json")
    ap.add_argument("--video_dir", required=True)
    ap.add_argument("--n", type=int, default=40)
    ap.add_argument("--steps", type=int, default=20)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    imgdir = os.path.join(a.out, "img"); os.makedirs(imgdir, exist_ok=True)
    import torch, torch.nn.functional as F
    if torch.cuda.is_available():
        torch.zeros(1, device="cuda")

    E2 = json.load(open(a.e2))["per_sample"]
    # 优先取 temporal_hit 的样本(生成给"选对的时刻", 展示最佳面貌; 不足再补)
    hits = [r for r in E2 if r.get("temporal_hit") and r.get("kf_ts") is not None]
    rest = [r for r in E2 if not r.get("temporal_hit") and r.get("kf_ts") is not None]
    picked = (hits + rest)[:a.n]
    print(f"[gen_from_e2] 取 {len(picked)} 条(其中时刻命中 {sum(1 for r in picked if r.get('temporal_hit'))})", flush=True)

    # ---- 抽关键帧 ----
    R = []
    for r in picked:
        vp = find_video(r["vid"], a.video_dir)
        if not vp:
            continue
        try:
            fr = read_frames(vp, fps=1.0)
            if not fr:
                continue
            kf = min(fr, key=lambda f: abs(f[0] - r["kf_ts"]))
            item = {"vid": r["vid"], "kf_ts": r["kf_ts"], "query": r.get("query", ""),
                    "object": r.get("object", ""), "temporal_hit": bool(r.get("temporal_hit"))}
            img = kf[1].convert("RGB")
            img.save(os.path.join(imgdir, f'{r["vid"]}_keyframe.png'))
            item["_kf"] = img
            R.append(item)
        except Exception as e:
            print("skip", r["vid"], str(e)[:80], flush=True)
    print(f"[frames] {len(R)} 关键帧就绪", flush=True)

    # ---- FLUX 生成 ----
    from diffusers import FluxKontextPipeline
    pipe = FluxKontextPipeline.from_pretrained(FLUX_DIR, torch_dtype=torch.bfloat16)
    pipe.enable_sequential_cpu_offload()
    try:
        pipe.vae.enable_slicing(); pipe.vae.enable_tiling()
    except Exception:
        pass
    print("[C] FLUX loaded", flush=True)
    for i, r in enumerate(R):
        subj = r["object"] or "the main person"
        instr = f"{subj}, {EDIT}"
        r["instruction"] = instr
        try:
            g = pipe(image=r["_kf"], prompt=instr, guidance_scale=2.5, num_inference_steps=a.steps).images[0]
            g.save(os.path.join(imgdir, f'{r["vid"]}_generated.png'))
            r["_gen"] = g
        except Exception as e:
            r["err"] = str(e)[:120]; r["_gen"] = None
        print(f"  gen {i+1}/{len(R)}", flush=True)
    del pipe
    import gc; gc.collect(); torch.cuda.empty_cache()

    # ---- CLIP 双分 ----
    from transformers import CLIPModel, CLIPProcessor
    cm = CLIPModel.from_pretrained(CLIP, use_safetensors=True).to("cuda").eval()
    cp = CLIPProcessor.from_pretrained(CLIP)
    ss, ta = [], []
    for r in R:
        if r.get("_gen") is None:
            continue
        try:
            inp = cp(text=[r["instruction"]], images=[r["_kf"], r["_gen"]],
                     return_tensors="pt", padding=True, truncation=True).to("cuda")
            with torch.no_grad():
                o = cm(**inp)
            ie = F.normalize(o.image_embeds, dim=-1); te = F.normalize(o.text_embeds, dim=-1)
            r["clip_subject_sim"] = round(float((ie[0]*ie[1]).sum()), 4)
            r["clip_text_align"] = round(float((ie[1]*te[0]).sum()), 4)
            ss.append(r["clip_subject_sim"]); ta.append(r["clip_text_align"])
        except Exception as e:
            r["err_clip"] = str(e)[:80]
    summary = {"n": len(R), "n_generated": sum(1 for r in R if r.get("_gen") is not None),
               "mean_clip_subject_sim": round(sum(ss)/len(ss), 4) if ss else None,
               "mean_clip_text_align": round(sum(ta)/len(ta), 4) if ta else None,
               "edit_prompt": EDIT, "selector": "E2 two-stage"}
    for r in R:
        r.pop("_kf", None); r.pop("_gen", None)
    json.dump({"summary": summary, "per_sample": R}, open(os.path.join(a.out, "results.json"), "w", encoding="utf-8"), ensure_ascii=False, indent=2)
    print("=== SUMMARY ===", flush=True); print(json.dumps(summary, ensure_ascii=False, indent=2), flush=True)
    print("GEN_FROM_E2_DONE", flush=True)


if __name__ == "__main__":
    main()
