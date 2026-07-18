# D档全量: 40对关键帧, 内容锁死+艺术拉满(与gen_d.py同prompt)
import os, glob
os.environ.setdefault("HF_HOME", "/root/autodl-tmp/hf")
os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")
import torch
from PIL import Image
PROMPT=("Transform this exact frame into a striking cinematic movie poster. "
        "STRICTLY KEEP the same people, same faces, same clothing, same poses, and same scene layout - do not add or remove anything. "
        "But completely re-light and re-grade it like a blockbuster poster: golden-hour rim lighting, teal-and-orange cinematic color grade, "
        "high contrast, glossy premium finish, atmospheric depth, dramatic yet bright poster look. "
        "Subjects must remain clearly visible and well-lit. No text, no watermark.")
src="/root/autodl-tmp/gen_e2_40/img"; out="/root/autodl-tmp/gen_d40/img"
os.makedirs(out, exist_ok=True)
kfs=sorted(glob.glob(os.path.join(src,"*_keyframe.png")))
from diffusers import FluxKontextPipeline
pipe=FluxKontextPipeline.from_pretrained("/root/autodl-tmp/flux_kontext", torch_dtype=torch.bfloat16)
pipe.enable_sequential_cpu_offload()
try: pipe.vae.enable_slicing(); pipe.vae.enable_tiling()
except Exception: pass
for i,kf in enumerate(kfs):
    vid=os.path.basename(kf).replace("_keyframe.png","")
    if os.path.exists(os.path.join(out, f"{vid}_generated.png")):
        print(f"d40 {i+1}/{len(kfs)} skip", flush=True); continue
    img=Image.open(kf).convert("RGB")
    g=pipe(image=img, prompt=PROMPT, guidance_scale=2.5, num_inference_steps=20).images[0]
    g.save(os.path.join(out, f"{vid}_generated.png"))
    Image.open(kf).save(os.path.join(out, f"{vid}_keyframe.png"))
    print(f"d40 {i+1}/{len(kfs)}", flush=True)
print("GEN_D40_DONE", flush=True)
