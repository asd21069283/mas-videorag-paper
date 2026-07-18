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

### Abstract (draft — compress to venue limit before submission)

Existing video-language systems *describe* video in text (video RAG), *generate* media from text/image prompts, or — in recent any-to-any models — map video to a generated image without localization or faithfulness accountability; to our knowledge, none take a **video plus a textual instruction** and return an **interleaved image-text** answer in which each image is **newly generated from, and verifiably faithful to, an explicitly localized keyframe**. We propose a cooperative two-agent framework — an *Understand Agent* that localizes question-relevant evidence and a *Generation Agent* that produces grounded images — coupled by a **shared multimodal RAG** memory. Our core method is a **question-driven, object-level localization-to-generation pipeline**: subject-verb-object cues from the instruction drive keyframe selection, open-vocabulary grounding, and a generate-or-crop decision, with cross-modal consistency checks for hallucination control. We formalize the task and introduce **VKIG-Bench**, to our knowledge the first benchmark scoring *keyframe consistency* and *spatio-temporal localization* alongside text and image quality. A real-video prototype runs end-to-end on a single 24GB GPU. Diagnosing temporal keyframe selection as the dominant bottleneck (a temporal-oracle study), we introduce a **two-stage selector** — MLLM coarse temporal windowing followed by object-conditioned in-window selection — which lifts joint localization accuracy from 0.433 to **0.567** on a 282-clip HC-STVG v2 subset (separately, 0.627 with a denser grid, n=300) — outperforming a NeurIPS'25 zero-shot STVG method (0.447) on the same clips after bridging both outputs to our joint keyframe metric — and holds at scale (0.419→0.540, n=1000), training-free (no parameter training or adaptation). On the generation side, our faithfulness protocol (human + MLLM-judge) exposes what CLIP similarity misses — only 30% of permissively-prompted generations depict the true moment (n = 40) — and shows faithfulness is instruction-controllable (10%→100% in a matched A/B, n = 10; the best faithfulness-artistry tier sustains 70% at n = 40), with the faithfulness–creativity trade-off mapped as the open frontier.

---

## 2. Introduction

**Motivation.** Much of today's content work consists of turning *moments* inside video into **usable, faithful visual assets**. A creator on a video platform needs a cover image of their clip's highlight — a cover that misrepresents the content is penalized as click-bait, so it must depict a *real* moment, yet a raw screenshot is rarely presentable. An enterprise turns screen recordings into **step-by-step illustrated manuals** — an inherently *interleaved image-text* artifact, one localized moment per step. A short-drama platform mass-produces posters and highlight art from episodes. In all of these the needed answer is a **picture plus a few words**, traceable to a real moment yet rendered for presentation: faithfulness here means **semantic consistency** — same subject, scene, and composition — not pixel-level identity recovery. (Identity-critical uses such as forensic review employ only the *localization* half of our pipeline and receive the original frame; generated imagery is never evidence.) Frontier MLLMs can watch video and *tell* you about it, but they neither localize precisely (quantified in §6.6) nor produce an accountable image; general text-to-image models produce images with no provenance in a source video. To our knowledge, few current systems close this loop end to end; a concurrent preprint [PVTG, arXiv:2607.12882] does close a video→faithful-thumbnail loop for the cover-image use case (preference-aware highlight retrieval → VLM-guided diffusion), but emits a single thumbnail with no explicit spatio-temporal localization protocol, no protocol-level faithfulness metric, and no instruction-driven interleaved answer (§3). Retrieval-augmented video question answering [VideoRAG, arXiv:2502.01549; Video-RAG, arXiv:2411.13093] reads long videos and answers **in text only** — it *tells* but does not *show*. Conversely, retrieval-augmented image generation [ImageRAG, arXiv:2502.09411; ORIG, arXiv:2510.22521; Gen-Searcher, arXiv:2603.28767] and interleaved image-text generators [M2RAG, arXiv:2411.16365; M2IO-R1, arXiv:2508.06328] *show*, but take **text (or image) input, never video**, and the displayed images are usually **retrieved/selected** rather than newly generated and grounded in a specific source frame.

**Gap.** We characterize the target task by four jointly-required properties: (a) **video input**, (b) output images that are **newly generated**, (c) **interleaved image-text** output, and (d) orchestration by a **multi-agent system (MAS) with shared multimodal RAG**. We are careful about what is and is not new here. *Any-to-any* models — X-VILA [arXiv:2405.19335] and NExT-GPT [arXiv:2309.05519] — can already map video input to a generated image, and X-VILA's visual-embedding highway even targets input-output visual consistency; unified generators [Emu3.5, arXiv:2510.26583; Show-o2, arXiv:2506.15564] produce interleaved image-text natively. **What, to our knowledge, no reviewed work provides is the conjunction we formalize**: question-driven, *explicit spatio-temporal localization* of the evidence (when/where in the video), image generation that is **verifiably faithful to the localized source keyframe** (a stated protocol and metric, not an emergent property), a long-form interleaved image-text answer, and a benchmark that scores this keyframe consistency — video-side multimodal RAG [VimRAG, arXiv:2602.12735; VideoRAG, arXiv:2501.05874] lacks (b)(c); interleaved generation/benchmarks [MRAMG-Bench, arXiv:2502.04176; M2IO-R1, arXiv:2508.06328] lack (a); any-to-any models lack the localization protocol, the faithfulness accountability, and (d). The most credible threat remains a *single unified* model bolted onto an off-the-shelf video RAG; we hypothesize that a MAS+RAG design is **not easily replaceable** by such a monolith because it provides **controllability, source traceability, and keyframe consistency** that a black-box end-to-end model does not expose — a claim our benchmark is designed to test; the unified-model comparison itself (BAGEL/Show-o2 bolted onto an off-the-shelf video RAG) has not been run and remains future work (planned baselines, §6.5).

**Contributions.**
- **A new task and benchmark.** We formalize *video-grounded interleaved image-text generation* and introduce **VKIG-Bench**, to our knowledge the first benchmark that scores **keyframe-consistency** (generated image vs. its source-keyframe object) and **spatio-temporal localization** (when/where the answer lives in the video), in addition to text correctness, image-text alignment, and image quality.
- **A cooperative MAS + shared-RAG framework.** An Understand Agent and a Generation Agent communicate through a *shared* multimodal RAG memory with two feedback loops (grounding→reselection, generation→memory write-back), unifying the understanding and generation ends that prior video MAS keep separate [MAGNET, arXiv:2506.07016; LVAS-Agent, arXiv:2503.10719; MovieAgent, arXiv:2503.07314; Mora, arXiv:2403.13248].
- **A two-stage temporal selector inside a question-driven, object-level localization-to-generation method.** A temporal-oracle study quantifies keyframe *timing* — not grounding — as the dominant bottleneck (§6.5); we address it by **adapting the coarse-to-fine selection paradigm** [UniTime, arXiv:2506.18883; Focus, arXiv:2510.27280] to keyframe selection for faithful generation — MLLM coarse temporal windowing → object-conditioned in-window selection (§4.3②) — lifting joint localization acc@0.3 from 0.433 to **0.567** on a 282-clip HC-STVG v2 subset and outperforming a bridged NeurIPS'25 zero-shot STVG baseline (0.447) on identical clips under our joint keyframe metric — training-free (no parameter training or adaptation) **[VERIFIED, n=282]**. The localized keyframe then drives open-vocabulary grounding and a generate-or-crop decision — beyond "select-only" interleaving [M2RAG, arXiv:2411.16365] — with cross-modal consistency checks [ScaleCap, arXiv:2506.19848].

---

## 3. Related Work

**Video multimodal RAG (text output).** VideoRAG [arXiv:2502.01549] builds cross-video knowledge graphs over hundreds of hours and answers with a single RAG pipeline; Video-RAG [arXiv:2411.13093] is a training-free plug-in that extracts OCR/ASR/open-vocabulary detections as auxiliary text. VimRAG [arXiv:2602.12735] extends multimodal RAG on the video side. All output **text**; their video front-ends are reusable for our Understand Agent, but none close the loop to a generated image. The distinction we draw: Video-RAG's detector yields boxes/keyframes that it only converts to text — we route the same signal into **image production**.

**Interleaved image-text generation and retrieval-augmented generation.** M2RAG [arXiv:2411.16365] retrieves web image-text and produces interleaved answers, but **selects** images rather than generating them. M2IO-R1 [arXiv:2508.06328] adds RL-based, trainable interleaved generation. MRAMG-Bench [arXiv:2502.04176] benchmarks multimodal answer generation. Retrieval-augmented *generation of new images* — ImageRAG [arXiv:2502.09411], ORIG [arXiv:2510.22521], Gen-Searcher [arXiv:2603.28767] — is the closest on the output side but is **text-input, no video**. Closest on the video→generated-image axis is a concurrent thumbnail-generation preprint [PVTG, arXiv:2607.12882]: preference-aware highlight retrieval selects a keyframe and VLM-guided diffusion renders a faithful thumbnail — a single cover image, with no bbox/temporal-IoU localization protocol, no protocolized faithfulness metric, and no interleaved answer. We inherit their interleaving/judging scaffolds and replace the image pool with **video-derived keyframe conditions**.

**Keyframe selection and grounded captioning.** AKS [arXiv:2502.21271] frames keyframe selection as maximizing question-relevance plus temporal coverage, training-free. VideoEspresso [arXiv:2411.14794] de-duplicates frames via BGE-M3 caption similarity and reasons over ~2 frames. GROVE [arXiv:2503.10781] generates captions while densely grounding mentioned objects via SVO triples. We combine these: **object-conditioned** question-driven selection (AKS×GROVE) with BGE-M3 redundancy pruning (VideoEspresso), then ground at the object level.

**Hallucination control and causal video understanding.** ScaleCap [arXiv:2506.19848] uses heuristic Q&A plus contrastive sentence scoring to remove hallucinated caption content; we lift this test-time idea to **cross-modal (image-text mutual-evidence) consistency** as a RAG factuality gate. CUVA [arXiv:2405.00181] reframes video anomaly understanding as What/Why/How with multi-dimensional metrics; we borrow its causal dimensions for evaluation.

**Video multi-agent systems and unified models.** Video MAS — MAGNET [arXiv:2506.07016] (RAG→scoring agents→meta-agent), LVAS-Agent [arXiv:2503.10719] (role-based dubbing crew), MovieAgent [arXiv:2503.07314] (director/writer agents with per-character LoRA), Mora [arXiv:2403.13248] (chained generation agents) — are largely **single-direction pipelines** with weak role asymmetry and no shared cross-end RAG; our two-agent design is differentiated by **bidirectional feedback and a shared memory**. Unified image-text models BAGEL [arXiv:2505.14683] and Show-o2 [arXiv:2506.15564] motivate our controllability/traceability argument: they could approximate the pipeline end-to-end but do not expose keyframe-grounded, source-traceable outputs or a way to verify them (untested here; a unified-model control is on the planned-baseline list, §6.5). Likewise, the any-to-any models conceded in §2 (X-VILA, NExT-GPT, Emu3.5) map video to generated images without an explicit localization protocol or a faithfulness metric.

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

**Notation:**

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

**② Two-stage temporal grounding and keyframe selection** *(the paper's core selection contribution; verified §6.6)*. Frame-wise similarity scoring — whether query-level `s(Q,F)` [AKS, arXiv:2502.21271] or our object-level `s(o,F)` — measures *appearance*, not *action timing*: it fires whenever the subject is visible, which on person-centric video is most of the clip. We therefore split temporal localization from appearance scoring:

*Stage A — coarse temporal window (MLLM).* Sample a sparse grid `G = {F_1..F_g}` (uniform, `g=10` frames) and ask the video-MLLM `M` (Qwen3-VL) to localize the queried event over the ordered grid:

```
W = M( G, Q )  →  [t_s, t_e]        (grid indices → time window, half-cell dilation at borders)
```

This exploits the MLLM's cross-frame reasoning — which frame-wise embedding similarity fundamentally lacks — at the cost of one extra MLLM call per clip. Coarse-to-fine temporal selection is an established paradigm in long-video *question answering* [UniTime, arXiv:2506.18883; Focus, arXiv:2510.27280; HiMu, arXiv:2603.18558; LeAdQA, arXiv:2507.14784]; our contribution is not the paradigm but its **diagnosis-driven adaptation to keyframe selection for faithful generation** — the oracle study (§6.5) localizes the pipeline's bottleneck to timing, and this adaptation resolves a large share of it (§6.6). Training-free; the gold interval is never revealed to `M` (sanity-checked, §6.6).

*Stage B — object-conditioned selection within `W`.* Restricted to frames in `W`, score with a z-normalized two-channel mixture of query and object relevance and take the temporally-smoothed peak (this is the configuration verified as E2 in §6.6):

```
s(o, F) = sim( ψ_img(F), ψ_txt(o) )
S(F) = (1−w_o)·ŝ(Q,F) + w_o·ŝ(o,F),   F ∈ W,  w_o = 0.6     (ŝ = z-scored)
F* = argmax_{F∈W} smooth(S(F))
```

A three-channel variant adding an *action-phrase* channel `ŝ(a,F)` (weights `w_q,w_o,w_a = 0.3/0.4/0.3`, untuned defaults) is evaluated separately as E1 in §6.6; combining it with Stage A is future work (no factorial run yet).

*Design context (not fully enabled in the verified runs):* BGE-M3 caption pruning [VideoEspresso, arXiv:2411.14794] before Stage A, and AKS-style coverage `c(I)` when a budget `B>1` of keyframes is requested; the coverage form below is our instantiation (AKS publishes no closed form):

```
I* = argmax_{I⊆W, |I|≤B} [ Σ_{F∈I} S(F) + λ·c(I) ],   c(I) = − Σ_j | n_j(I) − |I|/M |
```

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

MRAMG-Bench [arXiv:2502.04176] and similar interleaved-generation benchmarks evaluate factual image generation but have **no video and no keyframe-provenance consistency**; V-STaR [Author, arXiv:2503.11495] has video + bounding boxes but scores QA, not *generated* images. **VKIG-Bench unifies video provenance + generated image + keyframe consistency** — a slot that, to our knowledge, existing benchmarks do not fill. This is the moat against unified models [BAGEL, arXiv:2505.14683; Show-o2, arXiv:2506.15564]: it directly measures whether displayed images are *traceable to and consistent with* a source frame.

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

### 6.3 Dev-set results — V-STaR, our system **[VERIFIED]**

We ran the full pipeline (object-conditioned keyframe selection → Qwen3-VL grounding → SDXL-Turbo generation) over the **first 150 videos from the V-STaR test split, used here as a development subset**, on a single RTX 4090D 24GB (**1 primary keyframe/clip**; 0–1000 grounding). All 150 completed.

**Localization is reported *jointly* (temporal ∧ spatial) — the honest metric.** A keyframe counts as correct only if its timestamp falls in the gold temporal window **and** its box matches the gold box (IoU ≥ τ). We also report a *diagnostic* spatial-only number (box vs. the nearest-in-time gold box), which is **optimistic** because it ignores whether the moment is right.

| Metric (n = 150) | Value |
|---|---|
| **Localization acc (temporal-hit ∧ IoU ≥ 0.3)** | **0.233** |
| **Localization acc (temporal-hit ∧ IoU ≥ 0.5)** | **0.213** |
| Temporal recall (keyframe ts ∈ gold window) | **0.26** |
| **Spatial IoU when temporally correct** | **0.72** |
| CLIP subject sim (gen vs. keyframe) | 0.66 |
| CLIP text align (gen vs. instruction) | 0.29 |
| *diag:* mean spatial IoU (nearest-frame) | 0.39 |

**Honest reading — the bottleneck is *when*, not *where*.** When the pipeline picks a keyframe at the right moment, grounding is accurate (**IoU 0.72**); but only **26%** of keyframes land in the gold window, so end-to-end localization is **0.23**. The open problem is *temporal keyframe selection*, not the detector or box regression.

**Keyframe-selection ablation (50 videos, Qwen grounding).** Object-conditioned scoring (frames scored by target-*object* presence, not whole-question relevance) mainly improves the *quality* of the chosen frame:

| Selection | loc. acc@0.3 (joint) | IoU when temporally correct |
|---|---|---|
| Query-relevance | 0.24 | 0.69 |
| + temporal smoothing | 0.22 | 0.56 |
| **+ object-conditioned (ours)** | **0.28** | **0.79** |

**Grounding head-to-head (50 videos, identical keyframes: query-relevance + smoothing selection).** On matched keyframes, Qwen3-VL native grounding beats Grounding-DINO-base on the diagnostic spatial IoU (**0.323 vs. 0.292**) but *not* on the joint metric (acc@0.3 **0.22 vs. 0.24**) — its boxes are better, but grounding cannot fix timing. (The 0.28 in the ablation table above uses object-conditioned *selection*, i.e., different keyframes, and is not a same-keyframe grounding comparison.) We adopt Qwen3-VL grounding for its box quality and single-model economy; stronger G-DINO-large / DINO-X (API/less-available) remain future work.

*Implementation note (MVP).* The runs above use CLIP 1 fps frame scoring with z-normalized object conditioning and temporal smoothing for selection, Qwen3-VL for grounding, and SDXL-Turbo (V-STaR) for generation; the design components BGE-M3 pruning, recursive AKS coverage, and shared-RAG feedback (§4) are **not fully enabled** in this run. *Frame-reading cap (disclosed):* the 1-fps reader caps at 64 frames, i.e. the first ~64 s of each clip; on the small fraction of V-STaR clips longer than this whose gold window falls after 64 s, a hit is structurally impossible, so the V-STaR numbers here are a mild **lower bound** (removing the cap can only raise them). HC-STVG v2 clips are 20 s and never reach the cap, so §6.4–6.6 are unaffected; part of the V-STaR→HC-STVG temporal-recall gap in §6.4 is therefore attributable to this reader cap rather than to the data alone. Numbers span V-STaR 150 and HC-STVG 200–1000 (per-section n as stated); the bridged external baseline is in §6.6 and the FLUX-Kontext faithfulness study in §6.7.

### 6.4 Cross-dataset main results — HC-STVG v2 (test) **[VERIFIED, n = 1000]**

To test whether the "*when*, not *where*" finding is V-STaR-specific or general, we run the same localization pipeline on **HC-STVG v2** — an independent, human-annotated spatio-temporal grounding benchmark of 20 s clips with per-frame person tubes and a natural-language description. We evaluate on **1000 video-ready samples** from the community HF mirror (`ShijianW01/hc-stvg2`, test split with gold tubes; 6 samples skipped for missing videos). Grounding target = the noun phrase before the caption's first verb (HC-STVG `sub` field as fallback). Same joint metric; **temporal hit is strict (pad = 0 s), consistent with the V-STaR `batch_eval.py` numbers above** (the generic `eval/spatiotemporal_iou.py` must be run with `--t_pad 0` to match this strict setting; its default is 0.5 s); spatial gold = the tube box nearest in time to the predicted keyframe (appropriate for moving-person tubes). Results are stable across scale (n = 200 → 1000 move acc@0.3 0.430 → 0.419).

| Metric (HC-STVG v2 test, **n = 1000**) | Value | (V-STaR, n = 150) |
|---|---|---|
| **Localization acc (temporal-hit ∧ IoU ≥ 0.3)** | **0.419** | 0.233 |
| **Localization acc (temporal-hit ∧ IoU ≥ 0.5)** | **0.396** | 0.213 |
| Temporal recall (keyframe ts ∈ gold window) | **0.524** | 0.26 |
| **Spatial IoU when temporally correct** | **0.664** | 0.72 |
| *diag:* temporal recall (±0.5 s pad) | 0.587 | — |
| *diag:* mean spatial IoU (nearest-frame) | 0.564 | 0.39 |

**Cross-dataset reading — the finding generalizes, and the temporal gap shrinks on well-scoped clips.** On HC-STVG's 20 s single-action clips, temporal recall roughly **doubles** (0.52 vs. V-STaR's 0.26) and end-to-end localization nearly doubles (**0.42 vs. 0.23**), while box accuracy when temporally correct stays high (**0.66**). The bottleneck ordering is identical across both datasets — *temporal keyframe selection*, not grounding — indicating the bottleneck is not benchmark-specific under our evaluated datasets and grounding setup. (A few points of the recall gap are an instrument artifact: the 64-frame reading cap truncates long V-STaR clips but never the 20 s HC-STVG clips, §6.3; the ordering and conclusion are unchanged since HC-STVG's advantage only widens without the cap.) **Conservative note:** on HC-STVG our heuristic noun-phrase extractor still yields an over-long grounding phrase (> 8 words) in ~29% of samples (293/1000), which *depresses* these numbers; a proper parser/LLM extractor is expected to raise them.

**Metric convention (disclosed).** The joint accuracies use `n = 1000` as denominator; a sample whose grounding returns *no box* has no IoU and is counted as **incorrect** (it does not inflate acc). The diagnostic "*spatial IoU when temporally correct*" is averaged only over temporal hits that **did** produce a box — grounding returns a box on **96.6 %** of temporal hits, so this excludes 3.4 % (18/524); if the no-box cases were scored as IoU = 0 the value would be 0.641 instead of 0.664. The headline localization accuracies are unaffected by this choice.

*(Produced by `pipeline/hcstvg_eval.py`; **numbers independently recomputed from raw per-sample data in multiple verification passes** — a 45-agent audit re-derived every §6.3–6.7 figure from the per-sample JSON with zero discrepancy. The strict temporal convention (pad = 0) matches the generic `eval/spatiotemporal_iou.py` when run with `--t_pad 0`; note that the two scripts use different spatial-gold conventions (nearest-in-time tube box here vs. a fixed reference box there), so they agree on the temporal-hit gating but are not expected to match box-for-box — the headline joint accuracies are defined by `hcstvg_eval.py`. n = video-ready samples from the mirror subset; the full 1901-clip test split is fetched and a full-split run is straightforward.)*

### 6.5 Selection-strategy baselines — isolating the temporal bottleneck **[VERIFIED, n = 300]**

To quantify *how much* of the localization score is due to content-based keyframe selection versus a positional prior, we hold grounding fixed (Qwen3-VL) and swap only the selection step, on the same 300 HC-STVG test clips. `random` = a seeded random frame (floor); `uniform` = the video-midpoint frame (a strong positional prior, since HC-STVG action windows are temporally centred); `middle_gt` = the frame nearest the *gold interval's* midpoint — a **temporal oracle** that upper-bounds what perfect selection would give.

| Selection (n = 300, grounding fixed) | Temporal recall | **Loc. acc@0.3 (joint)** | acc@0.5 | IoU when correct |
|---|---|---|---|---|
| `random` (floor) | 0.350 | 0.257 | — | 0.643 |
| **Ours (object-conditioned CLIP)** | 0.563 | **0.443** | 0.423 | 0.655 |
| `uniform` (video midpoint, positional prior) | 0.557 | 0.440 | — | 0.667 |
| `middle_gt` (**temporal oracle**, upper bound) | 1.000 | **0.787** | — | 0.664 |

**Honest reading.** (1) Content selection clearly beats chance — **ours 0.443 vs. random 0.257** (+73%). (2) On HC-STVG, ours **ties** the trivial midpoint prior (0.443 vs. 0.440): because this dataset's action windows are centred, a positional prior is already strong, and our current content scoring does not yet beat it here (on street-scene V-STaR, object-conditioning *did* help — §6.3). (3) The decisive number is the **oracle: 0.787**. With grounding and box regression unchanged, *perfect temporal selection alone* would lift end-to-end localization from 0.44 to **0.79** — quantifying the open problem at **≈ +0.34 absolute**. This is the paper's central, load-bearing evidence that **temporal keyframe selection — not the detector — is the bottleneck**, and it is measured, not asserted.

*The bridged NeurIPS'25 zero-shot STVG comparison appears in §6.6; additional external system baselines and formal generation metrics (VQAScore/DreamSim/FID) remain future work. The comparison above isolates the selection component, which is where our diagnosis localizes the problem. Planned external baselines (all top-venue accepted, per advisor guidance 2026-07): **Zero-Shot STVG with MLLMs** [NeurIPS 2025, arXiv:2509.15178] — the primary head-to-head (same task, same datasets HC-STVG/VidSTG, same training-free MLLM route, code released); **LLaVA-ST** [CVPR 2025] and **VideoRefer Suite** [CVPR 2025] as trained-SOTA references; **Video-RAG** (visually-aligned) [NeurIPS 2025, arXiv:2411.13093] for the video-RAG side; **M2RAG** [SIGIR 2025, arXiv:2411.16365] for interleaved generation; and a **unified-model control** (BAGEL / Show-o2 bolted onto an off-the-shelf video RAG) to test the monolith-replaceability hypothesis (§2). Open-o3-Video (under review) and M2IO-R1/SpaceVLLM (arXiv-only) are cited but not used as baselines.*

### 6.6 Selection ablation — the two-stage selector **[VERIFIED, n = 282]**

Holding grounding fixed (Qwen3-VL) and evaluating on the **282-clip common subset** (the clips shared by our runs and the bridged baseline; identical denominators across all five rows, strict temporal pad = 0). §6.5's selector study uses the fuller 300-clip set; oracle values are reported per-denominator (0.787 @300, 0.780 @282).

**Bridging protocol (tube → keyframe).** The zero-shot STVG baseline emits a spatio-temporal tube; we convert it to a keyframe prediction: keyframe timestamp = the midpoint of its predicted temporal interval; predicted box = the tube box nearest that timestamp; gold box = the gold-tube box nearest the keyframe; then score with exactly our joint metric (strict pad = 0; no box = incorrect). The midpoint's frame rounding is computed both ways — floor/ceil: temporal recall 0.642/0.667, acc@0.3 0.447/0.443, acc@0.5 0.394/0.390. The table reports **floor**, the variant more favorable to the baseline on the headline acc@0.3; E2 wins under both roundings (`pipeline/hcstvg_compare.py`). Sanity anchor: the baseline's *native-protocol* reproduction in our environment (m_vIoU 0.221 over our full 299-clip reproduction run, before intersecting to the 282 common set) is consistent with the community-reported 0.207 (repo issue #4) and the paper's 0.236, so the bridged numbers rest on a faithful reproduction.

| Selection | Temporal recall | **acc@0.3 (joint)** | acc@0.5 |
|---|---|---|---|
| E0: object-conditioned scoring (§6.4 system) | 0.557 | 0.433 | 0.411 |
| E1: E0 + action-phrase channel | 0.610 | 0.489 | 0.468 |
| **E2: two-stage (MLLM window → in-window selection)** | **0.720** | **0.567** | **0.539** |
| (ref) NeurIPS'25 zero-shot STVG, bridged (protocol above; floor rounding) | 0.642 | 0.447 | 0.394 |
| (ref) temporal oracle upper bound | 1.000 | 0.780 | — |

**Reading.** (1) Both improvements outperform E0 and E2 is strongest (0.433→0.489→0.567, two independent modifications); box quality when temporally correct stays in a narrow band (0.61–0.64 across E0/E1/E2 on the 282 subset, no-box counted as IoU 0), so the improvement is **primarily associated with higher temporal recall** (conditional spatial IoU moves by ≤ 0.035, versus +0.16 temporal recall) — exactly where §6.5 located the bottleneck. Note E2 changes which keyframe is grounded, so this is an association, not a strictly controlled causal decomposition; E1 and E2 are two independent modifications (not a nested stack), and the full factorial ablation is future work. (2) **E2 outperforms the bridged NeurIPS'25 zero-shot STVG baseline by +0.121 acc@0.3 under our joint keyframe metric on the same clips** (bridged per the protocol above) while requiring no gradient-based test-time adaptation (it adds one MLLM call per clip — measured at **+1.46 s/clip**, §6.7 Cost). (3) The remaining oracle gap is 0.780−0.567 ≈ 0.21 (was ≈ 0.35 at the same 282 denominator): grid density, iterative window refinement, and audio/subtitle cues are natural next steps. **Exploratory leakage sanity check** (not a full audit): Stage A receives only the sparse frame grid and the query text — the gold interval is never passed in. Among the 203 temporally-correct predictions in the 282-clip subset, the median keyframe distance to the nearest gold boundary is 1.76 s (no boundary-snapping signature). Stage-A raw window outputs were not persisted in this run, so an end-to-end audit of the MLLM responses is future work.

**Scale confirmation [VERIFIED, n = 1000].** Re-running E2 (grid `g=10`) on 1000 HC-STVG test clips: temporal recall **0.688**, acc@0.3 **0.540**, acc@0.5 0.517 — versus the E0 system's 0.524/0.419/0.396 on the same 1000 clips (§6.4). The improvement holds at scale (mild regression from the 300-clip subset is expected).

**Stage-A grid density [VERIFIED, n = 300].** Densifying the sparse grid monotonically improves the window and the end metric — `g=10/15/20` → temporal recall 0.717/0.753/0.783, acc@0.3 **0.570/0.603/0.627** — shrinking the oracle gap to ≈0.16 at `g=20` (cost: proportionally more image tokens in the single Stage-A call). (Numbers are from the fixed selector: an earlier single-frame-window collapse bug silently reverted 10–13% of `g≥15` samples to the E0 baseline, mildly depressing those runs; `g=10` — and therefore every headline number — was unaffected, regression-verified bit-for-bit after the fix.) Saturation and cost-accuracy trade-off beyond `g=20` are TODO.

*Still planned:* A1 remove SVO conditioning; A2 whole-frame conditioning; A3 crop-only (=M2RAG); A4 BGE-M3 pruning; A5 shared RAG; A6 τ_g/τ_r sweep; A7 feedback loop; A8 `w` mixtures; A9 generator choice; A10 condition ablation; full factorial E1×E2; grid `g>20` saturation.

---

### 6.7 Keyframe-faithfulness protocol and results — generation **[VERIFIED, n = 40 + A/B n = 10]**

This section operationalizes the paper's faithfulness claim. **Protocol.** A generated image is *semantically faithful* to its source keyframe iff it depicts the *same moment*: same person(s), same clothing, same pose/action, same scene layout, and an emotional expression consistent with the source (crying→smiling is a violation; micro-level facial redraw without an emotion change is not — refined in Round 3); lighting/color/style changes are permitted (§2 motivation; identity-level pixel fidelity is explicitly out of scope, §7). We evaluate with (i) **human annotation** (authors; binary + unsure) and (ii) an **MLLM judge** (Qwen3-VL, two-image prompt asking strictly for a same-moment judgment, JSON output).

**Round 1 — baseline generation (n = 40).** FLUX.1-Kontext-dev conditioned on E2-selected keyframes with a *permissive* instruction ("re-render as a cinematic promotional poster shot, dramatic lighting"):

| Measure | Value |
|---|---|
| Human faithfulness rate | **12/40 (30%)** |
| CLIP subject similarity (same images) | 0.60 |
| MLLM judge faithfulness | 4/40 (10%) |
| Human–judge agreement (excl. unsure) | **31/39 (79.5%)**, with **zero false-accepts** (all 27 human-"unfaithful" caught) |

Failure modes (human-categorized): **(a) wholesale re-imagination** — the scene is replaced entirely (e.g., an outdoor group scene regenerated as a studio portrait of one man); **(b) clothing/identity drift**; **(c) over-darkening** (the "dramatic lighting" token over-executed). Two takeaways: *CLIP similarity (0.60) is blind to the 70% of pairs not rated faithful* — direct evidence that a dedicated faithfulness protocol is necessary; and the MLLM judge, while conservative, never passed an unfaithful image in this sample, making it a conservative *candidate* gate — held-out independent validation is required before large-scale use.

**Round 2 — instruction-constrained A/B (same 10 keyframes).** Replacing the instruction with an explicit preservation constraint ("enhance this exact frame…STRICTLY KEEP the same people, faces, clothing, poses and scene layout; only improve lighting/color/sharpness; keep bright; no text"):

| Instruction | Human faithful | MLLM judge |
|---|---|---|
| Permissive | 1/10 (10%) | 0/10 |
| **Constrained** | **10/10 (100%)** | 3/10 |

**In this matched sample, unfaithfulness behaves as an instruction-constraint problem rather than a hard capability ceiling** (n=10; capability limits at scale remain untested) — and our protocol is what makes this dimension measurable.

**The faithfulness–creativity trade-off (measured).** We added a second human-rated axis — *poster-likeness* (1 = looks like a screenshot, 2 = some polish, 3 = looks like a real poster) — and rated four instruction tiers on the same 10 keyframes (40 pairs, shuffled blind):

| Instruction tier | Human faithful | Poster-likeness (1–3) |
|---|---|---|
| A: permissive ("…promotional poster shot, dramatic lighting", abbrev.) | 1/10 (10%) | **2.50** |
| B: maximally constrained (preserve everything, enhance only) | **10/10 (100%)** | 1.10 |
| C: content locked, lighting/grading freed | **9/10 (90%)** | 1.20 |
| D: content locked, artistry maximized (cinematic re-grade, rim light, depth) | **9/10 (90%)** | **1.50** |

**Reading.** Once content is locked, poster-likeness rises monotonically as the stylistic vocabulary is pushed — B 1.10 → C 1.20 → D 1.50 — while faithfulness holds at 90–100%. So this is *not* a hard "faithfulness kills all artistry" wall: text instructions do recover some polish. But even maximal artistic instruction (D) plateaus at **1.50, well short of the 2.50** reached only when content is *unlocked* (A, at 10% faithfulness). Under single-instruction control, FLUX-Kontext lets us buy back partial artistry but cannot reach the faithful-*and*-striking sweet spot: the two axes compose only partway. This is a **bounded negative result for single-instruction control** (n=10, one annotator), and it is exactly what motivates **dual-channel conditioning** — identity/layout locked by an image condition, style freed by text (IP-Adapter/ControlNet-style, already part of our method design §4.3④) — as the designed path to the sweet spot (§7).

**Round 3 — the best tier at scale [VERIFIED, n = 40].** Applying the D-tier instruction to all 40 E2-selected keyframes (same keyframes and denominator as Round 1): human faithfulness **28/40 (70%)** vs. the permissive baseline's 12/40 (30%) — instruction control holds at scale, though below the 10-pair probe's 90% (small-n optimism). Poster-likeness averages **1.77** (n = 40; the 10-pair probe gave 1.50), confirming partial-artistry recovery. The annotation standard was sharpened for this round: an image whose **emotional expression contradicts the source moment** (e.g., crying → smiling) counts as unfaithful; micro-level facial redraw without an emotion change does not. CLIP subject similarity rises to **0.708** (vs. 0.601 permissive) — directionally consistent, but still blind to the 30% unfaithful tail. **Judge behavior shifts in this regime**: the MLLM judge passes 32/40, agrees with the human on 30/40 (75%), but **accepts 7 pairs the human rejected** — its zero-false-accept property (Round 1) does *not* transfer to the constrained regime, where residual failures are subtle (expression-level) rather than wholesale re-imagination. The judge is thus a candidate gate whose operating point is **regime-dependent**; independent held-out validation remains mandatory before any automated use.

**Cost (measured, single RTX 4090D, n=20 clips, per-sample averages).** Frame reading 1.07 s; E0 CLIP selection 0.34 s; E2 Stage-A adds **1.46 s** (one MLLM call over a 10-frame grid); grounding 0.90 s — E0 total ≈ 2.31 s vs. E2 ≈ 3.77 s per clip (one-time model loading: CLIP 16.3 s, Qwen3-VL 10.4 s). The two-stage selector's accuracy gain (§6.6) thus costs ~1.5 s per clip and no gradient computation, versus per-sample test-time *tuning* in the NeurIPS'25 baseline.

## 7. Conclusion and Limitations

**Conclusion.** We formalize *video-grounded interleaved image-text generation*, position it in a precisely-scoped gap, and propose a cooperative two-agent framework over a shared multimodal RAG with a question-driven, object-level localization-to-generation method and a keyframe-consistency-aware benchmark (VKIG-Bench). A temporal-oracle study isolates keyframe timing as the dominant bottleneck, and our **two-stage temporal selector** (MLLM windowing → in-window selection) lifts joint localization from 0.433 to 0.567 on a 282-clip HC-STVG v2 subset, outperforming a bridged NeurIPS'25 zero-shot STVG method on identical clips under our joint keyframe metric. A real-video prototype runs end to end on a single 24GB GPU.

**Limitations (honest).**
- **Temporal keyframe selection was the dominant bottleneck — partially addressed, gap remains.** *Before the two-stage selector*, temporal recall was **0.26 (V-STaR) / 0.52–0.56 (HC-STVG)**, capping joint localization at **0.23 / 0.42–0.44** even though the box is accurate once the moment is right (IoU 0.72 / 0.66). The two-stage selector (§4.3②, §6.6) lifts HC-STVG temporal recall to **0.720** and joint acc@0.3 to **0.567** (282-clip subset), but an oracle gap of **≈0.21** remains, and Stage A adds one MLLM call per clip (+1.46 s, measured §6.7). Future work: g>20 saturation sweep, iterative window refinement, audio/subtitle cues, a learned selector; the two-stage result is not yet re-verified on V-STaR. **[VERIFIED]**
- **Grounding & generation.** Qwen3-VL native grounding beats Grounding-DINO-base on box quality (same-keyframe diagnostic spatial IoU 0.323 vs. 0.292) but not on the joint metric (0.22 vs. 0.24, §6.3); stronger detectors (G-DINO-large / DINO-X, API-gated) are future work. The end-to-end V-STaR runs used SDXL-Turbo; the faithfulness study (§6.7) already uses the full FLUX-Kontext generator.
- **Scale, data provenance & generator.** Numbers are on dev subsets (V-STaR 150 / HC-STVG 200–1000; selector comparison on a 282-clip common subset) with SDXL-Turbo for the end-to-end runs (FLUX-Kontext used in the §6.7 faithfulness study), **not the full splits**; the only external baseline run so far is the bridged NeurIPS'25 zero-shot STVG (§6.6), whose full DSTH mode could not be reproduced from the official code (issue filed). HC-STVG videos are from a *community HF mirror* (`ShijianW01/hc-stvg2`), not the official package — we verified its annotation schema and gold tubes match the official format, but a re-run on the official release is on the list. The grounding-target noun phrase is a *heuristic* extractor (verb-boundary + `sub` fallback), over-long in ~29% of HC-STVG samples, which conservatively lowers the reported numbers. Scaling to full splits (HC-STVG 1901 / VKIG-Bench 300+→1–2k), the full FLUX-Kontext generator, proper metrics (VQAScore/DreamSim), and baselines are TODO.
- **Cross-domain generalization.** Validated on street-scene V-STaR and film-clip HC-STVG; short-drama (the target domain) and other domains remain a qualitative-demo plan, not yet quantified.
- **Human evaluation is single-annotator and small-n.** The faithfulness and dual-axis ratings (§6.7) come from one author annotator on n = 40/10/30 pairs with no inter-annotator agreement; the MLLM judge is validated only against those same labels. Independent multi-annotator re-labeling and a held-out judge validation are required before camera-ready or any large-scale use.
- **The faithfulness–creativity trade-off is measured but unresolved.** Under single-instruction control, locking content preserves faithfulness (70% at n = 40; 90–100% on 10-pair probes) while poster-likeness rises only partway as artistic instruction is pushed (B 1.10 → C 1.20 → D 1.50 on the probe, 1.77 at n = 40, §6.7) and stays below the 2.50 reached when content is unlocked. This bounded negative result (one annotator) motivates but does not yet validate **dual-channel conditioning** (image condition locks identity/layout, text frees style, §4.3④) — the designed path to the sweet spot remains unverified. A further instruction-level datapoint: adding a strong no-text constraint to the demo prompt visibly crowded out content preservation in one qualitative sample — single-instruction constraint budgets appear zero-sum.
- **Faithfulness is semantic, not forensic.** Our keyframe-consistency protocol measures *semantic* fidelity (subject/scene/composition); generative enhancement cannot recover identity-level detail absent from low-resolution sources (information-theoretically ill-posed; cf. face-hallucination failures and courtroom rejection of AI-enhanced evidence). The generation path is therefore scoped to creation-type outputs; identity-critical forensic scenarios receive localization + the original frame only. Multi-frame (burst) super-resolution, which can recover genuine sub-pixel information, is orthogonal future work.
- **Unverified design choices.** Coverage `c(I)` is our instantiation of AKS, not its published form; SVO extraction is a transfer of GROVE; several 2026 preprint references and the M2RAG metric set must be re-checked against source repos before camera-ready.

---

## References (placeholder keys — verify before submission)

- VideoRAG [arXiv:2502.01549]; Video-RAG [arXiv:2411.13093]; VimRAG [arXiv:2602.12735]
- M2RAG [arXiv:2411.16365]; M2IO-R1 [arXiv:2508.06328]; MRAMG-Bench [arXiv:2502.04176]
- ImageRAG [arXiv:2502.09411]; ORIG [arXiv:2510.22521]; Gen-Searcher [arXiv:2603.28767]
- AKS [arXiv:2502.21271]; VideoEspresso [arXiv:2411.14794]; GROVE [arXiv:2503.10781]
- UniTime [arXiv:2506.18883]; Focus [arXiv:2510.27280]; HiMu [arXiv:2603.18558]; LeAdQA [arXiv:2507.14784]
- Zero-Shot STVG with MLLMs [arXiv:2509.15178]; VideoRAG-over-corpus (Jeong et al.) [arXiv:2501.05874] — distinct from VideoRAG [arXiv:2502.01549]
- ScaleCap [arXiv:2506.19848]; CUVA [arXiv:2405.00181]
- MAGNET [arXiv:2506.07016]; LVAS-Agent [arXiv:2503.10719]; MovieAgent [arXiv:2503.07314]; Mora [arXiv:2403.13248]
- BAGEL [arXiv:2505.14683]; Show-o2 [arXiv:2506.15564]; X-VILA [arXiv:2405.19335]; NExT-GPT [arXiv:2309.05519]; Emu3.5 [arXiv:2510.26583]
- V-STaR [arXiv:2503.11495]; Open-o3-Video/STGR [arXiv:2510.20579]; PVTG (personalized video thumbnail generation) [arXiv:2607.12882]
- Qwen3-VL [arXiv:2511.21631]; OminiControl [arXiv:2411.15098]; VQAScore [project page]

---

## 【给作者的下一步清单】（中文·2026-07-17 全面体检后更新）

1. **最后一跑（实验冻结前）**：D 档 maximal-artistry 探针（脚本已在实例 `/tmp/gen_d.py`）→ 出 tradeoff 最终结论 → 最优档 40 对全量重跑 + 短剧 demo 重生成。
2. **统一模型对照（§2 假设的实证）**：BAGEL/Show-o2 + 现成 video-RAG 的 monolith 对照——引言已改为 "benchmark is designed to test"，跑出来后再把措辞改回实证。
3. **多标注者复标**：忠实性/双轴人评目前为单一作者标注（n=40/10/30），camera-ready 前需独立复标 + MLLM judge held-out 验证（§7 已如实声明）。
4. **官方数据与补充消融**：HC-STVG 官方包复跑（现为社区镜像）；g>20 饱和扫描；E1×E2 全因子；E2 在 V-STaR 上复验。
5. **正式生成指标与成本**：VQAScore / DreamSim / FID 接入；baseline 侧每样本 test-time tuning 耗时补测（汇报里承诺过，论文 Cost 段目前只有我方耗时）。
6. **引用滚动核对**：33 个占位号已联网核过 24 个全对；投稿前逐条再核；**每周重跑一次 "video → faithful image generation" 检索直到截稿**（7-14 已冒出 PVTG arXiv:2607.12882，本次已引并写好让步与区分）。
7. **写作阶段**：摘要压缩到会场字数上限；本中文清单等非投稿内容移出正稿；两轮 Codex 对抗审。节点：ICLR 2027 摘要 9/19、全文 9/24。

> 一句话汇报：论文位于 `paper/draft_v0.md`。2026-07-17 已做 45-agent 全面体检：§6.3–6.7 全部数字从原始 per_sample 数据独立重算一致；体检发现的 1 条 critical（§6.3 同帧对比数字张冠李戴）与全部 major/minor 已修复（见 git 记录）。剩余：D 档探针 → 实验冻结 → 写作。
