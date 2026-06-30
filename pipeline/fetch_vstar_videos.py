# fetch_vstar_videos.py
# 用 remotezip 从 V-STaR 远程大 zip(29G) 里只抽前 N 个样本的视频(不下整包)。
# 用法: python fetch_vstar_videos.py --n 10 --out /root/autodl-tmp/vstar
import os, json, argparse
os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")
REPO = "V-STaR-Bench/V-STaR"; RT = "dataset"
ZIPURL = "https://hf-mirror.com/datasets/V-STaR-Bench/V-STaR/resolve/main/vstar_videos.zip"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=10)
    ap.add_argument("--out", default="/root/autodl-tmp/vstar")
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    from huggingface_hub import hf_hub_download
    from remotezip import RemoteZip
    p = hf_hub_download(REPO, "V_STaR_test.json", repo_type=RT, local_dir=a.out)
    data = json.load(open(p, encoding="utf-8"))
    vids, seen = [], set()
    for s in data:
        v = str(s["vid"])
        if v not in seen:
            seen.add(v); vids.append(v)
        if len(vids) >= a.n:
            break
    vdir = os.path.join(a.out, "videos"); os.makedirs(vdir, exist_ok=True)
    got = 0
    with RemoteZip(ZIPURL) as z:
        names = z.namelist()
        for v in vids:
            cand = [nm for nm in names if os.path.splitext(os.path.basename(nm))[0] == v]
            if cand:
                z.extract(cand[0], vdir); got += 1
            else:
                print("zip里没找到:", v)
    print(f"[done] 抽出 {got}/{len(vids)} 个视频 -> {vdir}/videos/  (前{a.n}个唯一vid)")
    print("样例命令: python pipeline/vkig_pipeline.py --video %s/videos/%s.mp4 --query <Q> --object <O>" % (a.out, vids[0]))


if __name__ == "__main__":
    main()
