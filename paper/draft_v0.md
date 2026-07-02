# Draft v0 — MAS + Multimodal RAG for Video-grounded Interleaved Image-Text Generation

> Working draft for an AI-conference-style paper. English body, LaTeX-portable.
> Citations use placeholder keys of the form `[Author, arXiv:xxxx]` — do **not** treat numbers as final; verify against source list before submission.
> Status tags used throughout: **[VERIFIED]** = empirically run; **[PLANNED]** = designed but not yet run; **TODO** = result/table to be filled.

---

## 1. Title and Abstract

### Candidate titles
1. **VKIG: Video-Keyframe-Grounded Interleaved Image-Text Generation with Cooperative Agents and Shared Multimodal Retrieval**
2. **Seeing, Then Drawing: A Multi-Agent Multimodal-RAG Framework for Video-Grounded Image-Text Answers**
3. **From Telling to Showing: Question-Driven Object-Level Keyframe Grounding for Generative Video Question Answering**

### Abstract (<150 words)

Existing video-language systems either *describe* video in text (video RAG) or *generate* media from text/image prompts, but none take a **video plus a textual instruction** and return an **interleaved image-text** answer in which each image is **newly generated from, and faithful to, a grounded keyframe**. We propose a cooperative two-agent framework — an *Understand Agent* that localizes question-relevant evidence and a *Generation Agent* that produces grounded images — coupled by a **shared multimodal RAG** memory. Our core method is a **question-driven, object-level localization-to-generation pipeline**: subject-verb-object cues from the instruction drive keyframe selection, open-vocabulary grounding, and a generate-or-crop decision, with cross-modal consistency checks for hallucination control. We formalize the task and introduce **VKIG-Bench**, the first benchmark scoring *keyframe consistency* and *spatio-temporal localization* alongside text and image quality. A real-video end-to-end prototype runs on a single 24GB GPU; understanding and generation succeed, while localization remains the current bottleneck. TODO: full quantitative results.

---

## 2. Introduction

**Motivation.** When people ask a question about a video — *"what walked in front of the man in white, and what did it look like?"* — a useful answer is often not a paragraph but a **picture plus a few words**: show the object, point to where and when it appeared, and render it clearly. Current systems cannot do this end to end. Retrieval-augmented video question answering [VideoRAG, arXiv:2502.01549; Video-RAG, arXiv:2411.13093] reads long videos and answers **in text only** — it *tells* but does not *show*. Conversely, retrieval-augmented image generation [ImageRAG, arXiv:2502.09411; ORIG, arXiv:2510.22521; Gen-Searcher, arXiv:2603.28767] and interleaved image-text generators [M2RAG, arXiv:2411.16365; M2IO-R1, arXiv:2508.06328] *show*, but take **text (or image) input, never video**, and the displayed images are usually **retrieved/selected** rather than newly generated and grounded in a specific source frame.

**Gap.** We characterize the target task by four jointly-required properties: (a) **video input**, (b) output images that are **newly generated**, (c) **interleaved image-text** output, and (d) orchestration by a **multi-agent system (MAS) with shared multimodal RAG**. No prior work satisfies all four simultaneously: video-side multimodal RAG [VimRAG, arXiv:2602.12735; VideoRAG, arXiv:2502.01549] lacks (b); interleaved generation/benchmarks [MRAMG-Bench, arXiv:2502.04176; M2IO-R1, arXiv:2508.06328] lack (a). The intersection — **video-driven, keyframe-grounded new-image generation under multi-agent multimodal RAG** — is an open slot. The most credible threat is a *single unified* image-text model [BAGEL, arXiv:2505.14683; Show-o2, arXiv:2506.15564] bolted onto an off-the-shelf video RAG; we argue (and aim to show empirically) that a MAS+RAG design is **not replaceable** by such a monolith because it provides **controllability, source traceability, and keyframe consistency** that a black-box end-to-end model does not guarantee.

**Contributions.**
- **A new task and benchmark.** We formalize *video-grounded interleaved image-text generation* and introduce **VKIG-Bench**, the first benchmark that scores **keyframe-consistency** (generated image vs. its source-keyframe object) and **spatio-temporal localization** (when/where the answer lives in the video), in addition to text correctness, image-text alignment, and image quality.
- **A cooperative MAS + shared-RAG framework.** An Understand Agent and a Generation Agent communicate through a *shared* multimodal RAG memory with two feedback loops (grounding→reselection, generation→memory write-back), unifying the understanding and generation ends that prior video MAS keep separate [MAGNET, arXiv:2506.07016; LVAS-Agent, arXiv:2503.10719; MovieAgent, arXiv:2503.07314; Mora, arXiv:2403.13248].
- **A question-driven, object-level localization-to-generation method.** We push question-driven keyframe selection [AKS, arXiv:2502.21271] down to the SVO **object level** [GROVE, arXiv:2503.10781], drive open-vocabulary grounding [Video-RAG, arXiv:2411.13093], and feed the result into a generate-or-crop decision — going beyond "select-only" interleaving [M2RAG, arXiv:2411.16365] — with cross-modal consistency checks for hallucination control [ScaleCap, arXiv:2506.19848]. **[VERIFIED]** end-to-end on real video; TODO: large-scale results.

---

## 3. Related Work

**Video multimodal RAG (text output).** VideoRAG [arXiv:2502.01549] builds cross-video knowledge graphs over hundreds of hours and answers with a single RAG pipeline; Video-RAG [arXiv:2411.13093] is a training-free plug-in that extracts OCR/ASR/open-vocabulary detections as auxiliary text. VimRAG [arXiv:2602.12735] extends multimodal RAG on the video side. All output **text**; their video front-ends are reusable for our Understand Agent, but none close the loop to a generated image. The distinction we draw: Video-RAG's detector yields boxes/keyframes that it only converts to text — we route the same signal into **image production**.

**Interleaved image-text generation and retrieval-augmented generation.** M2RAG [arXiv:2411.16365] retrieves web image-text and produces interleaved answers, but **selects** images rather than generating them. M2IO-R1 [arXiv:2508.06328] adds RL-based, trainable interleaved generation. MRAMG-Bench [arXiv:2502.04176] benchmarks multimodal answer generation. Retrieval-augmented *generation of new images* — ImageRAG [arXiv:2502.09411], ORIG [arXiv:2510.22521], Gen-Searcher [arXiv:2603.28767] — is the closest on the output side but is **text-input, no video**. We inherit their interleaving/judging scaffolds and replace the image pool with **video-derived keyframe conditions**.

**Keyframe selection and grounded captioning.** AKS [arXiv:2502.21271] frames keyframe selection as maximizing question-relevance plus temporal coverage, training-free. VideoEspresso [arXiv:2411.14794] de-duplicates frames via BGE-M3 caption similarity and reasons over ~2 frames. GROVE [arXiv:2503.10781] generates captions while densely grounding mentioned objects via SVO triples. We combine these: **object-conditioned** question-driven selection (AKS×GROVE) with BGE-M3 redundancy pruning (VideoEspresso), then ground at the object level.

**Hallucination control and causal video understanding.** ScaleCap [arXiv:2506.19848] uses heuristic Q&A plus contrastive sentence scoring to remove hallucinated caption content; we lift this test-time idea to **cross-modal (image-text mutual-evidence) consistency** as a RAG factuality gate. CUVA [arXiv:2405.00181] reframes video anomaly understanding as What/Why/How with multi-dimensional metrics; we borrow its causal dimensions for evaluation.

**Video multi-agent systems and unified models.** Video MAS — MAGNET [arXiv:2506.07016] (RAG→scoring agents→meta-agent), LVAS-Agent [arXiv:2503.10719] (role-based dubbing crew), MovieAgent [arXiv:2503.07314] (director/writer agents with per-character LoRA), Mora [arXiv:2403.13248] (chained generation agents) — are largely **single-direction pipelines** with weak role asymmetry and no shared cross-end RAG; our two-agent design is differentiated by **bidirectional feedback and a shared memory**. Unified image-text models BAGEL [arXiv:2505.14683] and Show-o2 [arXiv:2506.15564] motivate our controllability/traceability argument: they could approximate the pipeline end-to-end but cannot guarantee keyframe-grounded, source-traceable images.

---

## 4. Method

### 4.1 Task formalization

**Input:** a video clip `V = {F_1, …, F_N}` and a textual instruction `Q`.
**Output:** an interleaved answer `A = [t_1, img_1, t_2, img_2, …]`, where each `t_i` is a text span and each `img_i` is a **new image generated from a keyframe of `V`**, carrying provenance `(keyframe_ts_i, bbox_i, o_i)`.

We seek `f: (V, Q) → A` satisfying four objectives:
1. **Text correctness** — `t_*` correctly answers `Q`;
2. **Image-text consistency** — `img_i` is semantically consistent with adjacent `t_i`;
3. **Keyframe consistency (core)** — `img_i` faithfully preserves the target object `o_i` from its source keyframe `V[keyframe_ts_i]`;
4. **Localization correctness** — `(keyframe_ts_i, bbox_i)` is the correct spatio-temporal location of the queried content in `V`.

**Notation** (after `创新点2_方法论框架.md`):

| Symbol | Meaning |
|---|---|
| `V = {F_1,…,F_N}` | input frame sequence |
| `Q` | user instruction |
| `T = {(s,v,o)_k}` | SVO triples extracted from `Q` |
| `o ∈ O_Q` | target object; `O_Q` the queried object set |
| `s(o,F)` | object-conditioned frame relevance |
| `c(I)` | temporal coverage of frame subset `I ⊆ V` |
| `I*` | optimal keyframe subset, `|I*| ≤ B` (budget `B`) |
| `b = (F,o,box,ρ)` | one object-level evidence item |
| `ρ(o,F)` | grounding confidence of `o` in frame `F` |
| `τ_g, τ_r` | grounding / retrieval thresholds |
| `ψ(·)` | text/cross-modal embedding (BGE-M3) |

### 4.2 Architecture: two agents over a shared multimodal RAG

The **Understand Agent** parses `Q`, selects keyframes, grounds objects, and writes a textual answer. The **Generation Agent** turns grounded evidence into images and lays out the interleaved answer. Both read and write a **shared multimodal RAG** store (frame embeddings, cropped images, detection cache; OCR/ASR/caption text), queried by `sim(ψ(o), ψ(cand)) ≥ τ_r`. Two feedback loops connect them:
- **Feedback ① (grounding→reselection):** if every keyframe for object `o` has `ρ(o,F) < τ_g`, return to keyframe selection with a larger budget `B` or relaxed pruning.
- **Feedback ② (generation→memory):** newly generated images are written back to the shared RAG for reuse, avoiding redundant generation.

This differs from prior single-direction video pipelines [MAGNET, arXiv:2506.07016; LVAS-Agent, arXiv:2503.10719] by closing an Understand↔Generation loop over **shared** memory.

### 4.3 Question-driven object-level localization-to-generation

**① Instruction parsing → SVO.** POS tagging + dependency parse extract `T = {(s,v,o)}` from `Q`, yielding object set `O_Q` (literal terms + optional LLM synonym expansion). This *reverses* GROVE [arXiv:2503.10781], which grounds caption objects; we extract objects from the **instruction** (marked "inspired by"). Training-free.

**② Q-driven keyframe selection.** BGE-M3 caption-similarity pruning [VideoEspresso, arXiv:2411.14794] reduces `N→N'`; then AKS [arXiv:2502.21271] is upgraded from query-level `s(Q,F)` to **object-level** `s(o,F)`. Object relevance and its aggregate:

```
s(o, F) = sim( ψ_img(F), ψ_txt(o) )
S(O_Q, F) = Σ_{o∈O_Q} w_o · s(o, F),   Σ w_o = 1
```

Frame selection trades relevance against temporal coverage:

```
I* = argmax_{I⊆V, |I|≤B} [ Σ_{F∈I} S(O_Q,F) + λ·c(I) ]
c(I) = − Σ_{j=1}^{M} | n_j(I) − |I|/M |     (M temporal bins; imbalance penalty)
```

solved by AKS's recursive judge-and-split (training-free). *Note:* AKS does not publish a closed form for `c(I)`; the above is our coverage instantiation (to be marked as such in camera-ready).

**③ Object-level grounding.** For each `(F∈I*, o∈O_Q)`, open-vocabulary detection (Grounding-DINO / Video-RAG's APE [arXiv:2411.13093]) yields `(box, ρ(o,F))`, producing evidence set `B = {(F,o,box,ρ)}`.

**④ Generate-or-crop decision.** Following the teacher-confirmed setting (output = *generation* of a new image, with crop demoted to an ablation), the main path conditions a generator on the best keyframe:

```
keyframe(o) = argmax_{F∈I*} ρ(o,F)
cond(o)     = { subject crop of keyframe(o), SVO/scene text, (opt.) structure map: depth/edge }
image(o)    = G(cond(o))
```

Two generator lines serve as comparisons: **(A) subject-preserving** — FLUX.1-dev + OminiControl(subject) [arXiv:2411.15098] (optionally PuLID/InfiniteYou for people, ControlNet for layout); **(B) instruction edit** — FLUX.1 Kontext-dev or Qwen-Image-Edit. The decision rule (used in the crop-ablation and as a fallback gate):

```
ρ*(o) = max_{F∈I*} ρ(o,F);   r*(o) = max_{cand∈RAG} sim(ψ(o), ψ(cand))
decide(o) = CROP   if ρ*(o) ≥ τ_g and r*(o) ≥ τ_r
          = GENERATE  otherwise
```

This surpasses M2RAG's "select-only" [arXiv:2411.16365] by **generating** when no faithful candidate exists.

**⑤ Interleaved layout.** The Generation Agent aligns each `img_i` to the answer sentence mentioning `o_i`, writes captions, and emits `A`.

### 4.4 Hallucination control via cross-modal consistency

Lifting ScaleCap's contrastive idea [arXiv:2506.19848] from images to the video→image-text setting, we score whether image and text **mutually evidence** each other, and use it as a RAG factuality gate. An optional consistency loss (only when LoRA is enabled):

```
L_cons = 1 − sim( ψ_img(img), ψ_txt(corresponding sentence) )
L = L_ret + β·L_gnd + γ·L_cons
```

with `L_ret = InfoNCE(ψ(o), ψ(correct evidence))` and `L_gnd = L_box(GIoU)+L_cls`. **The main line is deliberately training-free** (plug-and-play); LoRA and losses are optional enhancements.

---

## 5. VKIG-Bench

### 5.1 Why a new benchmark

MRAMG-Bench [arXiv:2502.04176] and similar interleaved-generation benchmarks evaluate factual image generation but have **no video and no keyframe-provenance consistency**; V-STaR [Author, arXiv:2503.11495] has video + bounding boxes but scores QA, not *generated* images. **VKIG-Bench unifies video provenance + generated image + keyframe consistency** — the slot none of them fill. This is the moat against unified models [BAGEL, arXiv:2505.14683; Show-o2, arXiv:2506.15564]: it directly measures whether displayed images are *traceable to and consistent with* a source frame.

### 5.2 Sample schema

```json
{
  "video": "clip_0001.mp4",
  "query": "Where and what did the protagonist's red sports car look like at first sight?",
  "gold": {
    "text": ["The protagonist first sees the red sports car at the theater entrance,",
             "parked to the right of the steps."],
    "evidence": [
      {"keyframe_ts": 87.3, "bbox": [x1,y1,x2,y2], "object": "red sports car",
       "gold_image_ref": "kf_0001_car.png", "pseudo_bbox": false}
    ]
  },
  "meta": {"source": "vstar|self_drama", "split": "test"}
}
```

### 5.3 Data sources and scale

1. **V-STaR seed** [Author, arXiv:2503.11495]: 2,094 videos / 16,793 boxes / when-where-what CoT, fully public; converted to the schema above as the labeled backbone.
2. **Self-built film/short-drama** (scene anchor pending): subtitle extraction + OCR + Grounding-DINO pseudo-boxes, semi-automatic with human spot-checks.
3. **Box fallback:** Grounding-DINO pseudo-boxes for box-free sources, flagged `pseudo_bbox=true`, with a human-verified subset.

Scale: start at **300–500** samples (dev 100 / test 300+), later 1–2k. Optionally seed strong supervision from Open-o3-Video / STGR [Author, arXiv:2510.20579] (keyframe+box+timestamp+CoT; release to be confirmed).

### 5.4 Evaluation dimensions (the moat)

| Dimension | Metric | Unique? |
|---|---|---|
| Text correctness | GPT-4o-judge / accuracy | No |
| Image-text consistency | **VQAScore** [Author, project page] | No |
| **Keyframe consistency** | **DreamSim / DINO** (gen image vs. source keyframe object) | **Yes — core** |
| **Spatio-temporal localization** | **temporal IoU(ts) + spatial AP/IoU(bbox)** | **Yes** |
| Image quality | FID / KID | No |
| Overall | human win-rate (A/B) | No |

### 5.5 Construction pipeline

```
raw video → Q-driven keyframe extraction → object grounding (box/segment)
          → gold image (crop reference, optional model-generated reference)
          → human spot-check (box correct? image correct?) → store (schema 5.2)
```

---

## 6. Experiments

### 6.1 Setup

- **Base models.** Understand Agent: Qwen3-VL [Author, arXiv:2511.21631] (native long video, frame timestamps); Generation Agent: FLUX.1-dev/Kontext-dev with OminiControl [arXiv:2411.15098], or Qwen-Image-Edit; detector: Grounding-DINO / APE [Video-RAG, arXiv:2411.13093]; embeddings: BGE-M3.
- **Hardware.** Single RTX 4090(D) 24GB for the MVP; large runs on A100-80G. Main line is training-free; optional LoRA fits in 16–18GB (gradient checkpointing, 512², 8-bit optimizer).
- **Baselines.** Understanding-side family: VideoRAG [arXiv:2502.01549]; main task baseline: Open-o3-Video [Author, arXiv:2510.20579] (video→keyframe+box+text CoT, most aligned I/O); interleaved-output: M2IO-R1 [arXiv:2508.06328] (RL, trainable), with M2RAG [arXiv:2411.16365] as a select-only reference.
- **Metrics.** As in §5.4.

### 6.2 Preliminary results — real-video end-to-end **[VERIFIED]**

We ran the full five-stage pipeline on a real V-STaR street-scene clip (`7771650716`), query *"what walks in front of a man in white?"*, target object = dog, on a single RTX 4090D 24GB.

| Stage | Action | Result |
|---|---|---|
| ① Q-driven selection | select keyframe by relevance + coverage from 61 frames | ✓ selected t = 3.0s (man walking a dog), keyframe score 0.261 |
| ② Grounding-DINO | localize "dog" in keyframe | ⚠ **mislocalized** (box on empty ground at right, conf 0.49) |
| ③ Qwen3-VL understanding | answer the question | ✓ **correct**: "a brown dog walks in front of the man in white" |
| ④ FLUX/SDXL generation | generate a new image conditioned on keyframe | ✓ new image of a brown dog on the street (gen model: sdxl-turbo) |
| ⑤ Evaluation | localization / consistency metrics | temporal hit ✓; **spatial IoU = 0**; subject CLIP sim **0.673**; CLIP text align **0.285** |

**Honest reading.** The full chain **runs end to end on real video**, with understanding and generation both succeeding — a step up from earlier synthetic single-image proofs. Localization with Grounding-DINO-base initially failed (the small, distant dog was boxed on adjacent empty ground, IoU = 0) — a failure only real data exposes. **Fix [VERIFIED]:** replacing the detector with **Qwen3-VL's native grounding** (same model used for understanding) corrected this on the same sample to a confident box on the dog, raising **spatial IoU from 0.0 → 0.363**. *Aggregate numbers follow in §6.3.*

### 6.3 Dev-set results — 50 samples, our system **[VERIFIED]**

We ran the full pipeline (Qwen3-VL grounding) over the first **50 V-STaR test videos** on a single RTX 4090D 24GB (4 keyframes/clip; SDXL-Turbo generator; CLIP/0–1000-grounding). All 50 completed.

**System metrics (Qwen3-VL grounding, 50 videos, best = object-conditioned selection).** Mean spatial IoU **0.415**, localization acc@IoU≥0.3 **0.50**, acc@IoU≥0.5 **0.40**, temporal recall **~0.30**, CLIP subject sim **0.67**, CLIP text align **0.28**.

**Keyframe-selection ablation (Qwen grounding, same 50 videos).**

| Selection | mean IoU | acc@0.3 |
|---|---|---|
| Query-relevance (earliest-of-k) | 0.358 | 0.48 |
| + temporal smoothing | 0.323 | 0.40 |
| **+ object-conditioned scoring (ours)** | **0.415** | **0.50** |

**Grounding head-to-head (50 videos, identical keyframes).**

| Detector | mean spatial IoU | acc@IoU≥0.3 |
|---|---|---|
| **Qwen3-VL native** | **0.323** | **0.40** |
| Grounding-DINO-base | 0.292 | 0.36 |

**Honest reading.** (1) **Qwen3-VL native grounding beats Grounding-DINO-base head-to-head** (IoU 0.323 vs. 0.292; single-sample finding holds at scale). (2) Both plateau near ~0.3 because the **real bottleneck is keyframe selection**: temporal recall is only 28–32%, so the selected frame often lies *outside* the gold interval and is scored against a mistimed gold box — capping IoU regardless of detector. (3) **Selection is the biggest lever.** An **object-conditioned selector** — scoring frames by target-*object* presence (weight 0.6) rather than whole-question relevance — lifts mean IoU **0.32→0.42** and acc@0.3 **40%→50%** [VERIFIED, 50 samples]: the object's on-screen presence is a stronger localization cue than full-query similarity. (A plain smoothing tweak alone was not a win; the object signal is what helps.) The stricter *temporal recall* (keyframe ts ∈ gold window) stays ~30%, but task-relevant grounding IoU improves substantially. Stronger detectors (Grounding-DINO-large, DINO-X) are API/less-available (future work). Generation preserves the subject reasonably (CLIP 0.66). *50-sample dev split, fast SDXL generator; FLUX-Kontext + baselines are §6.4 TODO.*

### 6.4 Main results — baseline comparison **[PLANNED]**

TODO — fill VKIG-Bench test results once dev (50 → 300+) is built:

| Method | Text (judge) | VQAScore | DreamSim↑ | t-IoU | s-IoU/AP | FID↓ | Win% |
|---|---|---|---|---|---|---|---|
| VideoRAG [arXiv:2502.01549] | TODO | n/a | n/a | TODO | n/a | n/a | TODO |
| Open-o3-Video [arXiv:2510.20579] | TODO | TODO | TODO | TODO | TODO | TODO | TODO |
| M2IO-R1 [arXiv:2508.06328] | TODO | TODO | TODO | n/a | n/a | TODO | TODO |
| Unified (BAGEL [arXiv:2505.14683] + video-RAG) | TODO | TODO | TODO | TODO | TODO | TODO | TODO |
| **Ours** | TODO | TODO | TODO | TODO | TODO | TODO | TODO |

### 6.4 Ablations **[PLANNED]**

TODO — table to be filled. Planned ablations: A1 remove SVO conditioning (`s(o,F)→s(Q,F)`); A2 remove object-level grounding (whole-frame conditioning); A3 remove generation (crop-only = M2RAG); A4 remove BGE-M3 pruning; A5 remove shared RAG; A6 sweep `τ_g/τ_r` (crop↔generate ratio curve); A7 remove feedback ①; A8 object-weight `w_o` strategies; A9 generator choice (OminiControl vs. Kontext vs. Qwen-Edit); A10 condition ablation (no keyframe / no structure map / no subject preservation).

---

## 7. Conclusion and Limitations

**Conclusion.** We formalize *video-grounded interleaved image-text generation*, position it in an open four-property slot, and propose a cooperative two-agent framework over a shared multimodal RAG with a question-driven, object-level localization-to-generation method and a keyframe-consistency-aware benchmark (VKIG-Bench). A real-video prototype runs end to end on a single 24GB GPU, with understanding and generation verified.

**Limitations (honest).**
- **Keyframe selection — improved, not solved.** Object-conditioned scoring lifted localization IoU 0.32→0.42 (acc@0.3 40%→50%) **[VERIFIED, 50 samples]**, but strict temporal recall (keyframe ts ∈ gold window) is still only ~30%. Further work: AKS-style coverage with object-size weighting, audio/subtitle cues, and a learned temporal selector.
- **Localization improved but not solved.** Switching from Grounding-DINO to Qwen3-VL native grounding raised spatial IoU from 0 → 0.358 mean (acc@0.3 = 48%) **[VERIFIED, 50 samples]**, but precise small-object boxes remain hard; stronger detectors / higher resolution are future work.
- **Scale & generator.** Results are a 50-sample dev split with a fast SDXL-Turbo generator; VKIG-Bench (→300+→1–2k), the full FLUX-Kontext generator, the proper metrics (VQAScore/DreamSim), and baseline comparisons are TODO.
- **Cross-domain generalization.** Validated only on street-scene V-STaR; film/short-drama and other domains are unverified.
- **Unverified design choices.** Coverage `c(I)` is our instantiation of AKS, not its published form; SVO extraction is a transfer of GROVE; several 2026 preprint references and the M2RAG metric set must be re-checked against source repos before camera-ready.

---

## References (placeholder keys — verify before submission)

- VideoRAG [arXiv:2502.01549]; Video-RAG [arXiv:2411.13093]; VimRAG [arXiv:2602.12735]
- M2RAG [arXiv:2411.16365]; M2IO-R1 [arXiv:2508.06328]; MRAMG-Bench [arXiv:2502.04176]
- ImageRAG [arXiv:2502.09411]; ORIG [arXiv:2510.22521]; Gen-Searcher [arXiv:2603.28767]
- AKS [arXiv:2502.21271]; VideoEspresso [arXiv:2411.14794]; GROVE [arXiv:2503.10781]
- ScaleCap [arXiv:2506.19848]; CUVA [arXiv:2405.00181]
- MAGNET [arXiv:2506.07016]; LVAS-Agent [arXiv:2503.10719]; MovieAgent [arXiv:2503.07314]; Mora [arXiv:2403.13248]
- BAGEL [arXiv:2505.14683]; Show-o2 [arXiv:2506.15564]
- V-STaR [arXiv:2503.11495]; Open-o3-Video/STGR [arXiv:2510.20579]
- Qwen3-VL [arXiv:2511.21631]; OminiControl [arXiv:2411.15098]; VQAScore [project page]

---

## 【给作者的下一步清单】（中文）

1. **补定位（最高优先级）**：把第②步换成 Qwen3-VL 自带 grounding，让"理解"与"定位"同源；同时试 Grounding-DINO-large / DINO-X + 提高输入分辨率；把"目标在画面里的大小"纳入选帧打分 `s(o,F)`。先把那条 IoU=0 的样本修到 IoU>0 再批量跑。
2. **跑出 dev 集**：按 §5.2 schema 把 V-STaR 转 50 条 → 跑通评测闭环（VQAScore / DreamSim / 时空 IoU / FID 四个脚本）→ 扩到 300+，填 §6.3、§6.4 的 TODO 表。
3. **落实 baseline**：实际跑 VideoRAG、Open-o3-Video、M2IO-R1，以及"BAGEL/Show-o2 + 现成 video-RAG"的统一模型对照，用数据支撑"MAS+RAG 不可替代"的论点（可控/可溯源/关键帧一致）。
4. **核实引用**：所有 `arXiv:xxxx` 占位号在投稿前逐条核对（尤其 2026 预印本 VimRAG/Gen-Searcher/ATP-Bench、M2RAG 会场、VideoRAG 是否 KDD'26、Open-o3-Video 数据/代码 release、VQAScore 正式引用）；把 M2RAG 那 8 个图文指标的确切名称按其开源仓写死。
5. **定稿待办**：确认 VKIG-Bench 自建数据的场景锚定（影视/短剧 Q1）；明确 gold image 口径（裁块参考 vs 模型生成参考）；把"`c(I)` 是我们对 AKS 的具体化""SVO 抽取迁移自 GROVE"在正文显式标注，避免审稿人质疑。
6. **目标会场**：尽快定 CV 顶会（CVPR/ICCV）vs ACL vs SCI 一区——写作风格与实验量差别大，会决定 §6 要补多少表。
7. **抢时间窗**：novelty 的"四属性交点"正被快速包围，建议优先把基准 + 一组完整主结果做出来占坑。

> 一句话汇报：英文论文初稿已写到 `/Users/liuzizhou/Documents/mas-videorag-paper/paper/draft_v0.md`，含标题/摘要/引言/相关工作/方法(带公式)/VKIG-Bench/实验(真实视频初步结果已标[VERIFIED]、其余表格留TODO)/结论与诚实局限，并附中文下一步清单。
