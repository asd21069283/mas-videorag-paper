# image_metrics.py
# VKIG 评测·图像端三指标(需 GPU + 下模型, 本地未实跑, 语法已校验)。
#   - VQAScore : 生成图 ↔ 指令/文本 一致性 (ECCV2024, 比 CLIPScore 更懂组合/关系)
#   - DreamSim : 生成图 ↔ 溯源关键帧 主体一致性 (我们的核心维度)
#   - FID      : 生成图分布质量
# 依赖(任选所需):
#   pip install t2v-metrics            # VQAScore (github.com/linzhiqiu/t2v_metrics)
#   pip install dreamsim               # DreamSim (github.com/ssundaram21/dreamsim)
#   pip install pytorch-fid            # FID
# 用法:
#   python image_metrics.py vqascore --pairs pairs.jsonl   # 每行 {"image":..,"text":..}
#   python image_metrics.py dreamsim --pairs pairs.jsonl   # 每行 {"image":..,"ref":..}
#   python image_metrics.py fid --gen_dir gen/ --ref_dir ref/
import argparse, json, sys


def _load(path):
    return [json.loads(l) for l in open(path, encoding="utf-8") if l.strip()]


def run_vqascore(pairs):
    """图↔文一致性, 越高越好(0~1)。"""
    import t2v_metrics
    clip_flant5 = t2v_metrics.VQAScore(model="clip-flant5-xxl")  # 论文默认
    scores = []
    for p in pairs:
        s = clip_flant5(images=[p["image"]], texts=[p["text"]])
        scores.append(float(s.item()))
    return {"metric": "VQAScore", "n": len(scores), "mean": sum(scores) / len(scores) if scores else 0.0}


def run_dreamsim(pairs):
    """生成图与溯源关键帧的感知相似度。dreamsim 返回的是'距离'(越小越像),
    这里同时给 距离均值 与 相似度=1-距离。"""
    import torch
    from dreamsim import dreamsim
    from PIL import Image
    model, preprocess = dreamsim(pretrained=True)
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    dists = []
    for p in pairs:
        a = preprocess(Image.open(p["image"]).convert("RGB")).to(dev)
        b = preprocess(Image.open(p["ref"]).convert("RGB")).to(dev)
        with torch.no_grad():
            dists.append(float(model(a, b).item()))
    mean_d = sum(dists) / len(dists) if dists else 0.0
    return {"metric": "DreamSim", "n": len(dists), "mean_distance": mean_d, "mean_similarity": 1.0 - mean_d}


def run_fid(gen_dir, ref_dir):
    """两个图像目录的 FID(越低越好)。"""
    from pytorch_fid import fid_score
    import torch
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    fid = fid_score.calculate_fid_given_paths([gen_dir, ref_dir], batch_size=50, device=dev, dims=2048)
    return {"metric": "FID", "value": float(fid)}


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    q = sub.add_parser("vqascore"); q.add_argument("--pairs", required=True)
    d = sub.add_parser("dreamsim"); d.add_argument("--pairs", required=True)
    f = sub.add_parser("fid"); f.add_argument("--gen_dir", required=True); f.add_argument("--ref_dir", required=True)
    a = ap.parse_args()
    if a.cmd == "vqascore":
        res = run_vqascore(_load(a.pairs))
    elif a.cmd == "dreamsim":
        res = run_dreamsim(_load(a.pairs))
    else:
        res = run_fid(a.gen_dir, a.ref_dir)
    print(json.dumps(res, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
