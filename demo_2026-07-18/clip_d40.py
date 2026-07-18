# D40 批量 CLIP 主体一致性(与 gen_e2_40_results.json 的 mean_clip_subject_sim 同口径)
import os, json, glob
os.environ.setdefault("HF_HOME","/root/autodl-tmp/hf")
os.environ.setdefault("HF_ENDPOINT","https://hf-mirror.com")
import torch, torch.nn.functional as F
from PIL import Image
from transformers import CLIPModel, CLIPProcessor
m=CLIPModel.from_pretrained("openai/clip-vit-base-patch32", use_safetensors=True).to("cuda").eval()
p=CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
src="/root/autodl-tmp/gen_d40/img"; per={}
for kf in sorted(glob.glob(os.path.join(src,"*_keyframe.png"))):
    gen=kf.replace("_keyframe.png","_generated.png")
    if not os.path.exists(gen): continue
    vid=os.path.basename(kf).replace("_keyframe.png","")
    a=Image.open(kf).convert("RGB"); b=Image.open(gen).convert("RGB")
    inp=p(text=["x"], images=[a,b], return_tensors="pt", padding=True).to("cuda")
    with torch.no_grad(): out=m(**inp)
    ie=F.normalize(out.image_embeds, dim=-1)
    per[vid]=round(float((ie[0]*ie[1]).sum().item()),4)
mean=round(sum(per.values())/len(per),4)
json.dump({"summary":{"n":len(per),"mean_clip_subject_sim":mean},"per_sample":per},
          open("/root/autodl-tmp/clip_d40_results.json","w"), indent=2)
print("mean_clip_subject_sim", mean, "n", len(per)); print("CLIP_D40_DONE")
