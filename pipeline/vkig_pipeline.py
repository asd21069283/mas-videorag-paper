# vkig_pipeline.py
# 完整 Video+Text -> Image+Text 流水线 (下次开机即用)。一段真实视频 + 一个问题 ->
#   ① Q-driven 选帧  ② Grounding-DINO 定位目标  ③ Qwen3-VL 理解  ④ FLUX-Kontext 基于关键帧生成  ⑤ 评测
# 用法:
#   python vkig_pipeline.py --selftest                      # 纯逻辑自测(无需GPU/模型)
#   python vkig_pipeline.py --video x.mp4 --query "..." --object "red car" --out_dir out/
# ⚠️ 需 GPU。各模型按需加载并释放(24G 装不下全部同时在显存)。版本敏感处标了 ⚠️。
import os, json, argparse, gc
os.environ.setdefault("HF_HOME", "/root/autodl-tmp/hf")
os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")

QWEN = os.environ.get("QWEN_PATH", "/root/autodl-tmp/Qwen3-VL-4B-Instruct")
GDINO = os.environ.get("GDINO_ID", "IDEA-Research/grounding-dino-base")
CLIP = "openai/clip-vit-base-patch32"


def _free():
    import torch
    gc.collect(); torch.cuda.empty_cache()


# ========== ① Q-driven 关键帧选择 ==========
def read_frames(video_path, fps=1.0, max_frames=64):
    """用 decord 按 fps 抽帧, 返回 [(ts, PIL.Image), ...]。"""
    from decord import VideoReader
    from PIL import Image
    vr = VideoReader(video_path)
    vfps = float(vr.get_avg_fps()) or 25.0
    step = max(1, int(round(vfps / fps)))
    idx = list(range(0, len(vr), step))[:max_frames]
    out = []
    for i in idx:
        out.append((i / vfps, Image.fromarray(vr[i].asnumpy())))
    return out


def select_by_score(scores, timestamps, k=4, n_bins=4):
    """【纯逻辑·可自测】AKS式选帧: 在相关性与时序覆盖间折中。
    把时间轴分 n_bins 个桶, 轮流从每个桶里挑当前分最高且未选的帧, 直到选满 k。
    这样既偏好高相关帧, 又保证覆盖全片(不扎堆)。返回选中的下标(按时间排序)。"""
    assert len(scores) == len(timestamps)
    if len(scores) <= k:
        return list(range(len(scores)))
    t0, t1 = min(timestamps), max(timestamps)
    span = (t1 - t0) or 1.0
    bins = [[] for _ in range(n_bins)]
    for i, t in enumerate(timestamps):
        b = min(n_bins - 1, int((t - t0) / span * n_bins))
        bins[b].append(i)
    for b in bins:
        b.sort(key=lambda i: -scores[i])           # 桶内按相关性降序
    chosen, bi = [], 0
    while len(chosen) < k and any(bins):
        b = bins[bi % n_bins]
        if b:
            chosen.append(b.pop(0))
        bi += 1
        if all(len(x) == 0 for x in bins):
            break
    return sorted(chosen, key=lambda i: timestamps[i])


def clip_image_text_scores(frames, text):
    """每帧与 text 的 CLIP 相似度(相关性打分)。frames=[(ts,PIL)]。"""
    import torch, torch.nn.functional as F
    from transformers import CLIPModel, CLIPProcessor
    m = CLIPModel.from_pretrained(CLIP, use_safetensors=True).to("cuda").eval()
    p = CLIPProcessor.from_pretrained(CLIP)
    imgs = [f[1] for f in frames]
    inp = p(text=[text], images=imgs, return_tensors="pt", padding=True, truncation=True).to("cuda")
    with torch.no_grad():
        o = m(**inp)
    ie = F.normalize(o.image_embeds, dim=-1); te = F.normalize(o.text_embeds, dim=-1)
    scores = (ie @ te.T).squeeze(-1).tolist()
    del m; _free()
    return scores


def select_keyframes(frames, query, k=4):
    scores = clip_image_text_scores(frames, query)
    ts = [f[0] for f in frames]
    idx = select_by_score(scores, ts, k=k)
    return [(frames[i][0], frames[i][1], scores[i]) for i in idx]   # [(ts, PIL, score)]


# ========== ② Grounding-DINO 目标定位 ==========
def ground_object(image, object_phrase):
    """在一帧里定位 object_phrase, 返回最高分框 [x1,y1,x2,y2] 与置信度。"""
    import torch
    from transformers import AutoProcessor, AutoModelForZeroShotObjectDetection
    proc = AutoProcessor.from_pretrained(GDINO)
    model = AutoModelForZeroShotObjectDetection.from_pretrained(GDINO).to("cuda").eval()
    text = object_phrase.lower().strip()
    if not text.endswith("."):
        text += "."                                  # G-DINO 要求小写、句号结尾
    inp = proc(images=image, text=text, return_tensors="pt").to("cuda")
    with torch.no_grad():
        out = model(**inp)
    # ⚠️ 不同 transformers 版本参数名可能是 threshold/box_threshold, 报错就换名
    res = proc.post_process_grounded_object_detection(
        out, inp.input_ids, threshold=0.30, text_threshold=0.25,
        target_sizes=[image.size[::-1]])[0]
    box, conf = None, 0.0
    if len(res["scores"]):
        j = int(res["scores"].argmax())
        box = [float(v) for v in res["boxes"][j].tolist()]
        conf = float(res["scores"][j])
    del model; _free()
    return box, conf


# ========== ③ Qwen3-VL 理解 ==========
def understand(images, question):
    """对若干关键帧(或单图)做视频理解, 返回 {answer, subject}。images=[PIL]。"""
    import torch
    from transformers import AutoProcessor
    try:
        from transformers import AutoModelForImageTextToText as VLM
    except Exception:
        from transformers import Qwen3VLForConditionalGeneration as VLM
    from qwen_vl_utils import process_vision_info
    proc = AutoProcessor.from_pretrained(QWEN)
    model = VLM.from_pretrained(QWEN, torch_dtype="auto", device_map="cuda").eval()

    def ask(content_imgs, q):
        content = [{"type": "image", "image": im} for im in content_imgs] + [{"type": "text", "text": q}]
        msgs = [{"role": "user", "content": content}]
        text = proc.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
        ii, vi = process_vision_info(msgs)
        inp = proc(text=[text], images=ii, videos=vi, padding=True, return_tensors="pt").to("cuda")
        with torch.no_grad():
            g = model.generate(**inp, max_new_tokens=128, do_sample=False)
        return proc.batch_decode(g[:, inp.input_ids.shape[1]:], skip_special_tokens=True)[0].strip()

    answer = ask(images, question)
    subject = ask(images[:1], "Name the single main object the question is about, as a short phrase "
                              "(type + color only, e.g. 'a red sports car'). Output only the phrase.")
    del model; _free()
    return {"answer": answer, "subject": subject}


# ========== ④ FLUX-Kontext 基于关键帧生成 ==========
def generate_image(keyframe, instruction, use_flux=True):
    """以关键帧为底 + 指令生成新图。优先 FLUX.1-Kontext, 失败回退 SDXL-Turbo img2img。"""
    import torch
    if use_flux:
        try:
            from diffusers import FluxKontextPipeline
            pipe = FluxKontextPipeline.from_pretrained(
                "black-forest-labs/FLUX.1-Kontext-dev", torch_dtype=torch.bfloat16)
            pipe.enable_model_cpu_offload()          # 24G 必开, 否则 OOM
            img = pipe(image=keyframe.convert("RGB"), prompt=instruction,
                       guidance_scale=2.5, num_inference_steps=28).images[0]
            del pipe; _free()
            return img, "flux-kontext"
        except Exception as e:
            print("[generate] FLUX 失败, 回退 SDXL:", str(e)[:160])
    from diffusers import AutoPipelineForImage2Image
    pipe = AutoPipelineForImage2Image.from_pretrained(
        "stabilityai/sdxl-turbo", torch_dtype=torch.float16, variant="fp16").to("cuda")
    pipe.set_progress_bar_config(disable=True)
    img = pipe(prompt=instruction, image=keyframe.convert("RGB"),
               num_inference_steps=6, strength=0.6, guidance_scale=0.0).images[0]
    del pipe; _free()
    return img, "sdxl-turbo"


# ========== ⑤ 评测(CLIP 主体一致 + 图文对齐) ==========
def clip_eval(keyframe, generated, instruction):
    import torch, torch.nn.functional as F
    from transformers import CLIPModel, CLIPProcessor
    m = CLIPModel.from_pretrained(CLIP, use_safetensors=True).to("cuda").eval()
    p = CLIPProcessor.from_pretrained(CLIP)
    inp = p(text=[instruction], images=[keyframe.convert("RGB"), generated.convert("RGB")],
            return_tensors="pt", padding=True, truncation=True).to("cuda")
    with torch.no_grad():
        o = m(**inp)
    ie = F.normalize(o.image_embeds, dim=-1); te = F.normalize(o.text_embeds, dim=-1)
    del m; _free()
    return {"clip_subject_sim": round(float((ie[0] * ie[1]).sum().item()), 4),
            "clip_text_align": round(float((ie[1] * te[0]).sum().item()), 4)}


# ========== 主流程: 跑一个样本 ==========
def run_one(video, query, object_phrase, out_dir, gold=None):
    from PIL import Image
    import torch
    # ⚠️ 先抢占建立 CUDA 上下文, 再用 decord 读帧。
    # 否则 decord 初始化会破坏后续 CUDA init, 报 "random_device could not be read"。
    if torch.cuda.is_available():
        torch.zeros(1, device="cuda")
    os.makedirs(out_dir, exist_ok=True)
    frames = read_frames(video, fps=1.0)
    print(f"[1] 抽帧 {len(frames)} 张")
    kfs = select_keyframes(frames, query, k=4)
    ts, kf_img, score = kfs[0]                        # 取相关性最高的关键帧做生成底图
    kf_img.save(os.path.join(out_dir, "keyframe.png"))
    print(f"[1] 选中关键帧 ts={ts:.1f}s score={score:.3f} (+{len(kfs)-1}张备选)")
    box, conf = ground_object(kf_img, object_phrase)
    print(f"[2] 定位 '{object_phrase}': box={box} conf={conf:.3f}")
    u = understand([k[1] for k in kfs], query)
    print(f"[3] 理解 answer={u['answer'][:80]!r} subject={u['subject']!r}")
    subj = u["subject"] if 0 < len(u["subject"]) < 60 else object_phrase
    instruction = f"{subj}, {os.environ.get('VKIG_EDIT','re-render in a new cinematic scene at night')}"
    gen, gen_model = generate_image(kf_img, instruction, use_flux=os.environ.get("USE_FLUX","1")=="1")
    gen.save(os.path.join(out_dir, "generated.png"))
    print(f"[4] 生成({gen_model}) -> generated.png")
    metrics = clip_eval(kf_img, gen, instruction)
    print(f"[5] 评测 {metrics}")
    result = {"video": video, "query": query, "object": object_phrase,
              "keyframe_ts": ts, "keyframe_score": round(score,4),
              "grounding": {"bbox": box, "conf": round(conf,4)},
              "understand": u, "instruction": instruction, "gen_model": gen_model,
              "metrics": metrics}
    if gold:
        try:
            import sys; sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "eval"))
            from spatiotemporal_iou import score_sample
            pred_ev = {"keyframe_ts": ts, "bbox": box}
            result["localization"] = score_sample(pred_ev, gold, siou_thr=0.5)
        except Exception as e:
            result["localization_error"] = str(e)[:160]
    json.dump(result, open(os.path.join(out_dir, "result.json"), "w", encoding="utf-8"),
              ensure_ascii=False, indent=2)
    print("[done] ->", os.path.join(out_dir, "result.json"))
    return result


def selftest():
    # 纯逻辑: 高分但扎堆 vs 覆盖
    scores =     [0.9, 0.85, 0.8, 0.1, 0.1, 0.7]
    ts =         [0.0, 0.5,  1.0, 5.0, 6.0, 9.0]
    sel = select_by_score(scores, ts, k=3, n_bins=3)
    assert len(sel) == 3
    # 应覆盖到后段(t=9.0 那个分0.7的帧), 而不是只取前三个扎堆的
    assert 5 in sel, f"覆盖失败, 选了 {sel}"
    assert sel == sorted(sel, key=lambda i: ts[i])
    # 少于k直接全选
    assert select_by_score([0.1,0.2], [0,1], k=4) == [0,1]
    print("✅ vkig_pipeline selftest 通过: 选帧覆盖逻辑(高相关+不扎堆)")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--selftest", action="store_true")
    ap.add_argument("--video"); ap.add_argument("--query"); ap.add_argument("--object")
    ap.add_argument("--out_dir", default="out")
    a = ap.parse_args()
    if a.selftest:
        selftest()
    else:
        assert a.video and a.query and a.object, "需 --video --query --object"
        run_one(a.video, a.query, a.object, a.out_dir)
