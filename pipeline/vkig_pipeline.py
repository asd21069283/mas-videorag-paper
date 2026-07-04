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
    try:
        container = av.open(video_path)
    except Exception:
        return []                                         # 打不开(缺失/损坏) -> 空
    if not container.streams.video:
        container.close(); return []                       # 无视频流
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


def clip_image_text_scores(frames, texts):
    """每帧与 texts 的 CLIP 相似度。texts=str -> 返回 list[float](每帧一个);
    texts=[str,...] -> 返回 list[list[float]](每个text一行, 每行每帧一个)。一次加载模型。"""
    import torch, torch.nn.functional as F
    from transformers import CLIPModel, CLIPProcessor
    single = isinstance(texts, str)
    tl = [texts] if single else list(texts)
    m = CLIPModel.from_pretrained(CLIP, use_safetensors=True).to("cuda").eval()
    p = CLIPProcessor.from_pretrained(CLIP)
    imgs = [f[1] for f in frames]
    inp = p(text=tl, images=imgs, return_tensors="pt", padding=True, truncation=True).to("cuda")
    with torch.no_grad():
        o = m(**inp)
    ie = F.normalize(o.image_embeds, dim=-1); te = F.normalize(o.text_embeds, dim=-1)
    sims = ie @ te.T                      # [N_frames, T_texts]
    per_text = sims.T.tolist()            # [T][N]
    del m; _free()
    return per_text[0] if single else per_text


def smooth_scores(scores, win=3):
    """【纯逻辑·可自测】移动平均平滑, 抑制单帧相关性尖峰, 让"持续相关的片段"胜出
    —— 更贴近 gold 时间窗(单帧尖峰常落在窗外)。"""
    n = len(scores)
    if n <= 2 or win <= 1:
        return list(scores)
    h = win // 2
    return [sum(scores[max(0, i-h):min(n, i+h+1)]) / (min(n, i+h+1) - max(0, i-h)) for i in range(n)]


def _znorm(xs):
    """z-score 归一化(帧维度); 长度<2 或零方差时原样返回。"""
    if len(xs) < 2:
        return list(xs)
    m = sum(xs) / len(xs)
    sd = (sum((x - m) ** 2 for x in xs) / len(xs)) ** 0.5
    return list(xs) if sd == 0 else [(x - m) / sd for x in xs]


def combine_scores(q_scores, o_scores, w_obj=0.6, normalize=True):
    """【纯逻辑·可自测】问题相关性 + 目标物体相关性 的加权组合。
    物体"在不在画面"是判断"什么时候"最强的信号, 故默认给物体更高权重(w_obj=0.6)。
    normalize=True: 先对两路分各自 z-score, 再混合(消除两个CLIP文本余弦的尺度差)。
    ⚠️ w_obj 是可调超参(论文里作消融, 非普适常数)。"""
    if not o_scores:
        return list(q_scores)
    q = _znorm(q_scores) if normalize else q_scores
    o = _znorm(o_scores) if normalize else o_scores
    return [(1 - w_obj) * qi + w_obj * oi for qi, oi in zip(q, o)]


def select_keyframes(frames, query, object_phrase=None, k=4, w_obj=0.6):
    """返回 [(ts,PIL,score)], **第0个=主关键帧**, 其余为覆盖备选。
    有 object_phrase 时用"问题+物体"联合打分(改善时间命中: 选到"目标真出现"的时刻)。"""
    ts = [f[0] for f in frames]
    if object_phrase:
        qs, os_ = clip_image_text_scores(frames, [query, object_phrase])   # 一次算两个
        scores = combine_scores(qs, os_, w_obj=w_obj)
    else:
        scores = clip_image_text_scores(frames, query)
    sm = smooth_scores(scores, win=3)
    primary = max(range(len(sm)), key=lambda i: sm[i])      # 主关键帧=持续相关段峰值(问题+物体联合)
    cover = select_by_score(sm, ts, k=k)
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


def parse_qwen_bbox(text, W, H, assume="norm1000"):
    """【纯逻辑·可自测】从 Qwen 输出里抽 bbox_2d, 转像素 [x1,y1,x2,y2]。找不到返回 None。
    assume = 该调用方 prompt 约定的坐标制, 消解 <=图像尺寸的中段值的二义性:
      "norm1000"(默认, 我们的 prompt 明确要 0-1000) / "pixel" / "auto"(纯启发, 无 prompt 约定时用)。
    通用规则(任何 assume 都先跑): <=1.5 视作 0-1; 超过图像边界的值只可能是 0-1000 归一化。"""
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
    mx = max(abs(x1), abs(y1), abs(x2), abs(y2))
    bound = float(max(W, H))
    if mx <= 1.5:                                         # 0-1 归一化(无歧义)
        mode = "norm1"
    elif mx > 1000.0:                                     # 超过 1000: 只可能是像素(Qwen 极少这样)
        mode = "pixel"
    elif mx > bound:                                      # 超过图像边界但 <=1000: 只可能是 0-1000 归一化
        mode = "norm1000"
    else:                                                 # 中段(<=图像边界): 纯凭数值分不清, 用调用方约定
        mode = {"pixel": "pixel", "auto": "pixel"}.get(assume, "norm1000")
    if mode == "norm1":
        x1, x2 = x1 * W, x2 * W; y1, y2 = y1 * H, y2 * H
    elif mode == "norm1000":
        x1, x2 = x1 * W / 1000.0, x2 * W / 1000.0
        y1, y2 = y1 * H / 1000.0, y2 * H / 1000.0
    # mode == "pixel": 原样用
    cx = lambda v: min(max(v, 0.0), float(W)); cy = lambda v: min(max(v, 0.0), float(H))  # 裁到图像边界
    x1, x2, y1, y2 = cx(x1), cx(x2), cy(y1), cy(y2)
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
    if not frames:                                    # 空视频/打不开/无视频流: 干净退出, 不崩
        msg = f"读不到任何帧(视频损坏或路径错): {video}"
        print("[skip]", msg)
        result = {"video": video, "query": query, "object": object_phrase, "error": "no_frames"}
        json.dump(result, open(os.path.join(out_dir, "result.json"), "w", encoding="utf-8"),
                  ensure_ascii=False, indent=2)
        return result
    kfs = select_keyframes(frames, query, object_phrase=object_phrase, k=4)
    if not kfs:                                        # 帧非空但选帧仍拿不到(极端兜底)
        print("[skip] 选帧为空:", video)
        result = {"video": video, "query": query, "object": object_phrase, "error": "no_keyframe"}
        json.dump(result, open(os.path.join(out_dir, "result.json"), "w", encoding="utf-8"),
                  ensure_ascii=False, indent=2)
        return result
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
    # 0-1 归一化 / 像素坐标+越界裁剪 / z-norm
    assert parse_qwen_bbox('{"bbox_2d":[0.1,0.2,0.5,0.9]}', 640, 360) == [64.0, 72.0, 320.0, 324.0]
    assert parse_qwen_bbox('{"bbox_2d":[100,50,1200,400]}', 640, 360) == [100.0, 50.0, 640.0, 360.0]  # 像素+clamp
    # 中段像素框([100,50,500,300] 全<=640): 默认按 prompt 约定当 0-1000; 显式 assume="pixel" 才原样保留
    assert parse_qwen_bbox('{"bbox_2d":[100,50,500,300]}', 640, 360, assume="pixel") == [100.0, 50.0, 500.0, 300.0]
    assert parse_qwen_bbox('{"bbox_2d":[100,50,500,300]}', 640, 360) == [64.0, 18.0, 320.0, 108.0]  # 默认 norm1000
    assert _znorm([1, 1, 1]) == [1, 1, 1]
    _z = _znorm([0.0, 10.0]); assert abs(_z[0] + 1) < 1e-9 and abs(_z[1] - 1) < 1e-9
    # smooth_scores: 孤立尖峰(i=0) vs 持续相关段(i=4,5,6) -> 平滑后峰值应落在持续段
    sm = smooth_scores([1.0, 0, 0, 0, 1.0, 1.0, 1.0], win=3)
    assert max(range(len(sm)), key=lambda i: sm[i]) >= 4, f"平滑应选持续相关段, got {sm}"
    # combine_scores: 物体信号(权重0.6)把峰值拉向"物体出现"的帧
    c = combine_scores([0.9, 0.1, 0.1], [0.1, 0.1, 0.9], w_obj=0.6)
    assert max(range(len(c)), key=lambda i: c[i]) == 2, f"物体条件应把峰值移到物体出现帧, got {c}"
    assert combine_scores([0.5, 0.5], None) == [0.5, 0.5]      # 无物体退回问题分
    print("✅ vkig_pipeline selftest 通过: 选帧覆盖 + bbox解析 + 平滑 + 物体条件联合打分")


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
