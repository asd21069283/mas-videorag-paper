# duanju_demo.py — 短剧定性demo: 视频片段+问题 → E2两阶段选帧 → Qwen定位 → FLUX忠实生成。
# 输出: out/{tag}_keyframe.png, {tag}_generated.png, {tag}_meta.json (含定位框/时刻)
# 用法(每条一跑):
#   FLUX_DIR=... python pipeline/duanju_demo.py --video xx.mp4 --t0 0 --t1 180 \
#     --query "女主在门口与男主争执" --object "the woman in the doorway" --tag ep1_scene1 --out /root/autodl-tmp/duanju_demo
import os, json, argparse, re
os.environ.setdefault("HF_HOME", "/root/autodl-tmp/hf")
os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")
import sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from vkig_pipeline import read_frames, smooth_scores, combine_scores, parse_qwen_bbox
from hcstvg_eval import QWEN, CLIP

FLUX_DIR = os.environ.get("FLUX_DIR", "/root/autodl-tmp/flux_kontext")
EDIT = os.environ.get("VKIG_EDIT", "re-render as a cinematic short-drama promotional poster, dramatic lighting")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--video", required=True)
    ap.add_argument("--t0", type=float, default=0); ap.add_argument("--t1", type=float, default=1e9)
    ap.add_argument("--query", required=True); ap.add_argument("--object", required=True)
    ap.add_argument("--tag", required=True); ap.add_argument("--out", required=True)
    ap.add_argument("--grid_n", type=int, default=12); ap.add_argument("--steps", type=int, default=20)
    ap.add_argument("--skip_gen", action="store_true")
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    import torch, torch.nn.functional as F
    if torch.cuda.is_available():
        torch.zeros(1, device="cuda")

    frames = [f for f in read_frames(a.video, fps=1.0) if a.t0 <= f[0] <= a.t1]
    print(f"[{a.tag}] 片段帧数 {len(frames)} ({a.t0}-{min(a.t1, frames[-1][0] if frames else 0)}s)", flush=True)
    if len(frames) < 5:
        print("片段太短, 退出"); return

    # ---- Stage A: Qwen 粗定位 ----
    from transformers import AutoProcessor
    try:
        from transformers import AutoModelForImageTextToText as VLM
    except Exception:
        from transformers import Qwen3VLForConditionalGeneration as VLM
    from qwen_vl_utils import process_vision_info
    proc = AutoProcessor.from_pretrained(QWEN)
    qm = VLM.from_pretrained(QWEN, torch_dtype="auto", device_map="cuda").eval()
    def qask(imgs, q, max_new=64):
        msgs=[{"role":"user","content":[{"type":"image","image":im} for im in imgs]+[{"type":"text","text":q}]}]
        text=proc.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
        ii,vi=process_vision_info(msgs)
        inp=proc(text=[text], images=ii, videos=vi, padding=True, return_tensors="pt").to("cuda")
        with torch.no_grad():
            g=qm.generate(**inp, max_new_tokens=max_new, do_sample=False)
        return proc.batch_decode(g[:, inp.input_ids.shape[1]:], skip_special_tokens=True)[0]
    idxs=[round(i*(len(frames)-1)/(a.grid_n-1)) for i in range(min(a.grid_n,len(frames)))]
    idxs=sorted(set(idxs)); grid=[frames[i][1] for i in idxs]
    raw=qask(grid, f'These {len(grid)} frames are uniformly sampled from a video in time order (frame 1..{len(grid)}). In which frames is this happening: "{a.query}"? Answer ONLY JSON {{"start_frame": s, "end_frame": e}} with 1-based numbers.')
    m=re.search(r'"start_frame"\s*:\s*(\d+).*?"end_frame"\s*:\s*(\d+)', raw, re.S)
    lo, hi = 0, len(frames)-1
    if m:
        s_i=max(0,min(int(m.group(1))-1,len(idxs)-1)); e_i=max(s_i,min(int(m.group(2))-1,len(idxs)-1))
        cell=(idxs[1]-idxs[0]) if len(idxs)>1 else 1
        lo=max(0, idxs[s_i]-cell//2); hi=min(len(frames)-1, idxs[e_i]+cell//2)
    print(f"[A] 窗口帧 {lo}-{hi} (t={frames[lo][0]:.0f}-{frames[hi][0]:.0f}s) raw={raw[:60]!r}", flush=True)

    # ---- Stage B: CLIP 窗内精选 ----
    from transformers import CLIPModel, CLIPProcessor
    cm=CLIPModel.from_pretrained(CLIP, use_safetensors=True).to("cuda").eval()
    cp=CLIPProcessor.from_pretrained(CLIP)
    sub=frames[lo:hi+1]
    inp=cp(text=[a.query, a.object], images=[f[1] for f in sub], return_tensors="pt", padding=True, truncation=True).to("cuda")
    with torch.no_grad():
        o=cm(**inp)
    sims=F.normalize(o.image_embeds,dim=-1)@F.normalize(o.text_embeds,dim=-1).T
    per=sims.T.tolist()
    comb=combine_scores(per[0], per[1], w_obj=0.6); sm=smooth_scores(comb, win=3)
    ki=max(range(len(sm)), key=lambda i: sm[i])
    kf_ts, kf = sub[ki][0], sub[ki][1].convert("RGB")
    kf.save(os.path.join(a.out, f"{a.tag}_keyframe.png"))
    del cm
    print(f"[B] 关键帧 t={kf_ts:.0f}s", flush=True)

    # ---- 定位框 ----
    graw=qask([kf], f'Locate the {a.object} in this image. Output ONLY JSON {{"bbox_2d":[x1,y1,x2,y2]}} integers 0-1000, top-left origin.')
    box=parse_qwen_bbox(graw, kf.width, kf.height)
    del qm; import gc; gc.collect(); torch.cuda.empty_cache()
    print(f"[G] box={box}", flush=True)

    # ---- FLUX 生成 ----
    gen_path=None
    if not a.skip_gen:
        from diffusers import FluxKontextPipeline
        pipe=FluxKontextPipeline.from_pretrained(FLUX_DIR, torch_dtype=torch.bfloat16)
        pipe.enable_sequential_cpu_offload()
        try: pipe.vae.enable_slicing(); pipe.vae.enable_tiling()
        except Exception: pass
        img=pipe(image=kf, prompt=f"{a.object}, {EDIT}", guidance_scale=2.5, num_inference_steps=a.steps).images[0]
        gen_path=os.path.join(a.out, f"{a.tag}_generated.png"); img.save(gen_path)
        del pipe; gc.collect(); torch.cuda.empty_cache()
        print("[C] 生成完成", flush=True)
    json.dump({"tag":a.tag,"video":a.video,"query":a.query,"object":a.object,
               "kf_ts":kf_ts,"window_s":[frames[lo][0],frames[hi][0]],"bbox":box},
              open(os.path.join(a.out, f"{a.tag}_meta.json"),"w",encoding="utf-8"),ensure_ascii=False,indent=2)
    print(f"DUANJU_{a.tag}_DONE", flush=True)


if __name__ == "__main__":
    main()
