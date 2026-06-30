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
    """用 PyAV 按 fps 抽帧, 返回 [(ts, PIL.Image), ...]。
    ⚠️ 不用 decord——decord 会破坏 PyTorch CUDA 初始化(random_device / segfault)。"""
    import av
    container = av.open(video_path)
    stream = container.streams.video[0]
    step_t = 1.0 / fps
    out, next_t = [], 0.0
    for i, frame in enumerate(container.decode(stream)):
        t = float(frame.pts * stream.time_base) if frame.pts is not None else i * (1.0 / (float(stream.average_rate) or 25.0))
        if t + 1e-6 >= next_t:
            out.append((t, frame.to_image()))
            next_t += step_t
            if len(out) >= max_frames:
                break
    container.close()
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


def smooth_scores(scores, win=3):
    """【纯逻辑·可自测】移动平均平滑, 抑制单帧相关性尖峰, 让"持续相关的片段"胜出
    —— 更贴近 gold 时间窗(单帧尖峰常落在窗外)。"""
    n = len(scores)
    if n <= 2 or win <= 1:
        return list(scores)
    h = win // 2
    return [sum(scores[max(0, i-h):min(n, i+h+1)]) / (min(n, i+h+1) - max(0, i-h)) for i in range(n)]


def select_keyframes(frames, query, k=4):
    """返回 [(ts,PIL,score)], **第0个=主关键帧(平滑相关性峰值, 用于定位/生成)**, 其余为覆盖备选。"""
    scores = clip_image_text_scores(frames, query)
    ts = [f[0] for f in frames]
    sm = smooth_scores(scores, win=3)
    primary = max(range(len(sm)), key=lambda i: sm[i])      # 主关键帧=持续相关段的峰值(不取孤立尖峰/最早帧)
    cover = select_by_score(sm, ts, k=k)                    # 备选: 用平滑分做覆盖选帧
    order = [primary] + [i for i in sorted(set(cover) - {primary}, key=lambda i: ts[i])]
    return [(frames[i][0], frames[i][1], scores[i]) for i in order]


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


# ========== ③ Qwen3-VL 理解 (+自带 grounding 出框) ==========
import re as _re


def parse_qwen_bbox(text, W, H):
    """【纯逻辑·可自测】从 Qwen 输出里抽 bbox_2d, 按 0-1000 归一化转像素 [x1,y1,x2,y2]。
    Qwen3-VL grounding 坐标是相对 0-1000 (官方); 找不到返回 None。"""
    nums = None
    m = _re.search(r'"bbox_2d"\s*:\s*\[([^\]]+)\]', text)
    if m:
        nums = [float(x) for x in _re.findall(r'-?\d+\.?\d*', m.group(1))]
    if not nums:                                     # 退而求其次: 找第一个4元数组
        m2 = _re.search(r'\[\s*(-?\d+\.?\d*)\s*,\s*(-?\d+\.?\d*)\s*,\s*(-?\d+\.?\d*)\s*,\s*(-?\d+\.?\d*)\s*\]', text)
        if m2:
            nums = [float(x) for x in m2.groups()]
    if not nums or len(nums) < 4:
        return None
    x1, y1, x2, y2 = nums[:4]
    if max(abs(x1), abs(y1), abs(x2), abs(y2)) <= 1000:   # 0-1000 归一化 -> 像素
        x1, x2 = x1 * W / 1000.0, x2 * W / 1000.0
        y1, y2 = y1 * H / 1000.0, y2 * H / 1000.0
    return [min(x1, x2), min(y1, y2), max(x1, x2), max(y1, y2)]


def understand(images, question, ground_image=None, object_phrase=None):
    """视频理解, 返回 {answer, subject, bbox(可选)}。images=[PIL]。
    若给 ground_image+object_phrase, 用 Qwen 自带 grounding 在该图上对目标出框(0-1000->像素)。"""
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
            g = model.generate(**inp, max_new_tokens=160, do_sample=False)
        return proc.batch_decode(g[:, inp.input_ids.shape[1]:], skip_special_tokens=True)[0].strip()

    answer = ask(images, question)
    subject = ask(images[:1], "Name the single main object the question is about, as a short phrase "
                              "(type + color only, e.g. 'a red sports car'). Output only the phrase.")
    bbox, bbox_raw = None, None
    if ground_image is not None and object_phrase:
        bbox_raw = ask([ground_image],
                       f'Locate the {object_phrase} in this image. Output ONLY JSON like '
                       f'{{"bbox_2d": [x1,y1,x2,y2]}} with integer coordinates normalized to 0-1000, top-left origin.')
        bbox = parse_qwen_bbox(bbox_raw, ground_image.width, ground_image.height)
    del model; _free()
    return {"answer": answer, "subject": subject, "bbox": bbox, "bbox_raw": bbox_raw}


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
    ground = os.environ.get("GROUND", "qwen")        # 默认用 Qwen 自带 grounding(理解+定位同源)
    if ground == "gdino":
        box, conf = ground_object(kf_img, object_phrase)
        u = understand([k[1] for k in kfs], query)
    else:
        u = understand([k[1] for k in kfs], query, ground_image=kf_img, object_phrase=object_phrase)
        box, conf = u.get("bbox"), (1.0 if u.get("bbox") else 0.0)
    print(f"[2] 定位 '{object_phrase}' ({ground}): box={box} conf={conf}")
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
    # parse_qwen_bbox: 0-1000 归一化 -> 像素(W=640,H=360)
    b = parse_qwen_bbox('{"bbox_2d": [100, 200, 300, 400], "label": "dog"}', 640, 360)
    assert b == [64.0, 72.0, 192.0, 144.0], b
    # 无 bbox_2d 时退回找4元数组
    b2 = parse_qwen_bbox('the box is [500,500,1000,1000]', 640, 360)
    assert b2 == [320.0, 180.0, 640.0, 360.0], b2
    assert parse_qwen_bbox("no box here", 640, 360) is None
    # smooth_scores: 孤立尖峰(i=0) vs 持续相关段(i=4,5,6) -> 平滑后峰值应落在持续段
    sm = smooth_scores([1.0, 0, 0, 0, 1.0, 1.0, 1.0], win=3)
    assert max(range(len(sm)), key=lambda i: sm[i]) >= 4, f"平滑应选持续相关段, got {sm}"
    print("✅ vkig_pipeline selftest 通过: 选帧覆盖 + Qwen bbox 解析 + 平滑选帧(抑尖峰)")


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
