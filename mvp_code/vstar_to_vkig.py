# vstar_to_vkig.py
# 把 V-STaR 数据集(HF: V-STaR-Bench/V-STaR)转成本项目的 VKIG schema(见 docs_任务形式化与专属基准_草案.md §2.1)。
# V-STaR 每条字段(已联网核对 HF 数据卡, 2026-06):
#   vid, domain, fps, width, height, frame_count,
#   question, chain(CoT), object, answer,
#   temporal_question, timestamps=[start,end],
#   spatial_question, spatial_question_2,
#   bboxes=[{frame_index:int, timestamp:float, xmin,ymin,xmax,ymax:int}, ...]
# VKIG 输出: {video, query, gold:{text, evidence:[{object,interval,keyframe_ts,bbox,track,pseudo_bbox}]}, meta}
#
# 用法:
#   python vstar_to_vkig.py --in vstar.jsonl --out vkig.jsonl --video_dir /path/to/videos
#   python vstar_to_vkig.py --selftest      # 无需数据, 本地自测转换逻辑
import argparse, json, os


def _to_list(x):
    """V-STaR 某些字段在不同导出里可能是 list 或被转义的字符串, 容错解析。"""
    if x is None:
        return []
    if isinstance(x, (list, tuple)):
        return list(x)
    if isinstance(x, str):
        try:
            v = json.loads(x)
            return v if isinstance(v, list) else [v]
        except Exception:
            return []
    return [x]


def _norm_box(b):
    """{xmin,ymin,xmax,ymax} -> [xmin,ymin,xmax,ymax]，并保证 min<=max。"""
    xmin, ymin = float(b["xmin"]), float(b["ymin"])
    xmax, ymax = float(b["xmax"]), float(b["ymax"])
    return [min(xmin, xmax), min(ymin, ymax), max(xmin, xmax), max(ymin, ymax)]


def convert_one(s, video_dir=""):
    boxes = _to_list(s.get("bboxes"))
    track = []
    for b in boxes:
        if all(k in b for k in ("xmin", "ymin", "xmax", "ymax")):
            ts = float(b.get("timestamp", 0.0))
            track.append({"ts": ts, "bbox": _norm_box(b)})
    track.sort(key=lambda e: e["ts"])

    interval = _to_list(s.get("timestamps"))
    interval = [float(interval[0]), float(interval[1])] if len(interval) >= 2 else None

    # 关键帧 = 时间区间中点最近的那个框; 无 interval 则取 track 中位
    keyframe_ts, keyframe_bbox = None, None
    if track:
        if interval:
            mid = (interval[0] + interval[1]) / 2.0
            kf = min(track, key=lambda e: abs(e["ts"] - mid))
        else:
            kf = track[len(track) // 2]
        keyframe_ts, keyframe_bbox = kf["ts"], kf["bbox"]

    video = os.path.join(video_dir, str(s["vid"])) if video_dir else str(s["vid"])

    text = [str(s.get("answer", ""))]
    if s.get("chain"):
        text.append(str(s["chain"]))

    evidence = []
    if keyframe_bbox is not None:
        evidence.append({
            "object": str(s.get("object", "")),
            "interval": interval,
            "keyframe_ts": keyframe_ts,
            "bbox": keyframe_bbox,
            "track": track,
            "pseudo_bbox": False,
        })

    return {
        "video": video,
        "query": str(s.get("question", "")),
        "object": str(s.get("object", "")),                # 顶层 object 与 HC-STVG schema 统一
        "gold": {"text": text, "evidence": evidence},
        "meta": {
            "source": "vstar",
            "domain": s.get("domain", ""),
            "fps": s.get("fps"),
            "wh": [s.get("width"), s.get("height")],
            "temporal_question": s.get("temporal_question", ""),
            "spatial_question": s.get("spatial_question", ""),
        },
    }


def _load(path):
    txt = open(path, encoding="utf-8").read().strip()
    if not txt:
        return []
    if txt[0] == "[":            # JSON array
        return json.loads(txt)
    return [json.loads(l) for l in txt.splitlines() if l.strip()]   # JSONL


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in", dest="inp", required=True, help="V-STaR json/jsonl")
    ap.add_argument("--out", required=True, help="输出 VKIG jsonl")
    ap.add_argument("--video_dir", default="", help="视频目录(拼到 vid 前)")
    args = ap.parse_args()
    rows = _load(args.inp)
    n = 0
    with open(args.out, "w", encoding="utf-8") as f:
        for s in rows:
            if not s.get("bboxes"):     # 无框样本跳过(VKIG 需要定位)
                continue
            f.write(json.dumps(convert_one(s, args.video_dir), ensure_ascii=False) + "\n")
            n += 1
    print(f"[done] {len(rows)} -> {n} 条 VKIG  ({args.out})")


def selftest():
    sample = {
        "vid": "abc123.mp4", "domain": "Sports", "fps": 30.0, "width": 1280, "height": 720,
        "frame_count": 900, "question": "What is the person holding when they first appear?",
        "chain": "The person enters at 2s holding a tennis racket.", "object": "tennis racket",
        "answer": "a tennis racket", "temporal_question": "When does the person appear?",
        "timestamps": [2.0, 6.0], "spatial_question": "Where is the racket?",
        "bboxes": [
            {"frame_index": 60, "timestamp": 2.0, "xmin": 100, "ymin": 50, "xmax": 200, "ymax": 300},
            {"frame_index": 120, "timestamp": 4.0, "xmin": 110, "ymin": 55, "xmax": 210, "ymax": 305},
            {"frame_index": 180, "timestamp": 6.0, "xmin": 120, "ymin": 60, "xmax": 220, "ymax": 310},
        ],
    }
    out = convert_one(sample, video_dir="/data/videos")
    assert out["video"] == "/data/videos/abc123.mp4"
    assert out["query"].startswith("What is the person holding")
    ev = out["gold"]["evidence"][0]
    assert ev["object"] == "tennis racket"
    assert ev["interval"] == [2.0, 6.0]
    assert ev["keyframe_ts"] == 4.0, f"中点应取 4.0s 的框, 得 {ev['keyframe_ts']}"
    assert ev["bbox"] == [110.0, 55.0, 210.0, 305.0], ev["bbox"]
    assert len(ev["track"]) == 3 and ev["pseudo_bbox"] is False
    assert out["gold"]["text"][0] == "a tennis racket"
    # 测被转义字符串型字段
    s2 = dict(sample, bboxes=json.dumps(sample["bboxes"]), timestamps="[2.0, 6.0]")
    out2 = convert_one(s2)
    assert out2["gold"]["evidence"][0]["keyframe_ts"] == 4.0
    print("✅ vstar_to_vkig selftest 通过: 中点关键帧 / 框归一化 / track / 转义字段容错")


if __name__ == "__main__":
    import sys
    if "--selftest" in sys.argv:
        selftest()
    else:
        main()
