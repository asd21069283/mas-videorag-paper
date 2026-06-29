# download_vstar_subset.py
# 从 HF 下载 V-STaR 的标注 + 前 N 个样本的视频(子集, 不下全量 29.6G)。
# 用法: python download_vstar_subset.py --n 20 --out /root/autodl-tmp/vstar
# ⚠️ V-STaR 的具体文件布局以仓库实际为准; 本脚本自动探测标注文件并按其中的 vid 找视频。
import os, json, argparse, re
os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")
REPO = "V-STaR-Bench/V-STaR"; RT = "dataset"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=20)
    ap.add_argument("--out", default="/root/autodl-tmp/vstar")
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)
    from huggingface_hub import list_repo_files, hf_hub_download

    files = list_repo_files(REPO, repo_type=RT)
    print(f"仓库共 {len(files)} 文件")
    ann_files = [f for f in files if re.search(r"\.(json|jsonl)$", f) and "test" in f.lower()] \
        or [f for f in files if re.search(r"\.(json|jsonl)$", f)]
    if not ann_files:
        print("⚠️ 没找到 json/jsonl 标注, 仓库文件样例:", files[:20]); return
    ann = ann_files[0]
    print("标注文件:", ann)
    p = hf_hub_download(REPO, ann, repo_type=RT, local_dir=args.out)
    txt = open(p, encoding="utf-8").read().strip()
    data = json.loads(txt) if txt[0] == "[" else [json.loads(l) for l in txt.splitlines() if l.strip()]
    print(f"标注 {len(data)} 条, 取前 {args.n} 条")

    video_files = [f for f in files if re.search(r"\.(mp4|mkv|avi|webm|mov)$", f, re.I)]
    sub = data[:args.n]
    got = 0
    for s in sub:
        vid = str(s.get("vid") or s.get("video_id") or s.get("video") or "")
        cand = [f for f in video_files if os.path.splitext(os.path.basename(f))[0] == os.path.splitext(vid)[0]] \
            or [f for f in video_files if vid and vid in f]
        if cand:
            hf_hub_download(REPO, cand[0], repo_type=RT, local_dir=args.out)
            got += 1
        else:
            print("  未找到视频:", vid)
    json.dump(sub, open(os.path.join(args.out, "subset.json"), "w", encoding="utf-8"), ensure_ascii=False, indent=2)
    print(f"[done] 下了 {got}/{len(sub)} 个视频; 标注子集 -> {args.out}/subset.json")
    print("下一步: python vkig_pipeline.py --video <out>/<vid> --query <question> --object <object>")


if __name__ == "__main__":
    main()
