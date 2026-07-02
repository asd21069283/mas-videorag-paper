# hcstvg_to_vkig.py
# 把 HC-STVG v2 标注转成本项目 VKIG schema(见 docs_任务形式化与专属基准_草案.md §2.1)。
# HC-STVG v2 每条字段(已联网核对 GitHub tzhhhh123/HC-STVG):
#   caption: 目标人物的自然语言描述(既当query又当grounding文本)
#   bbox: 逐帧目标框数组, 每个 [x,y,w,h](左上+宽高); 从 st_frame 起
#   st_time / ed_time: gold 时间区间(秒);  st_frame: bbox数组对应的首帧号(从1起)
#   img_num: 总帧数 = fps*20 (所有视频20秒);  width/height
# VKIG: {video, query, object, gold:{text, evidence:[{object,interval,keyframe_ts,bbox,track,pseudo_bbox}]}, meta}
#
# 用法:
#   python hcstvg_to_vkig.py --in anno_v2/train.json --out hcstvg_vkig.jsonl --video_dir /path/to/videos
#   python hcstvg_to_vkig.py --selftest
import argparse, json, os


def xywh_to_xyxy(b):
    x, y, w, h = float(b[0]), float(b[1]), float(b[2]), float(b[3])
    return [x, y, x + w, y + h]


def convert_one(vid, s, video_dir=""):
    """vid=视频标识; s=该条标注dict。返回 VKIG 样本。"""
    caption = str(s.get("caption", "")).strip()
    boxes = s.get("bbox") or []
    img_num = float(s.get("img_num") or (len(boxes) or 1))
    fps = img_num / 20.0 if img_num else 25.0            # HC-STVG 视频固定 20 秒
    st_frame = int(s.get("st_frame", 1))
    st_time = float(s.get("st_time", 0.0)); ed_time = float(s.get("ed_time", 20.0))

    track = []
    for i, b in enumerate(boxes):
        if b is None or len(b) < 4:
            continue
        frame_idx = st_frame + i                          # 1-based
        ts = (frame_idx - 1) / fps                         # 秒
        track.append({"ts": round(ts, 3), "bbox": xywh_to_xyxy(b)})

    keyframe_ts, keyframe_bbox = None, None
    if track:
        mid = (st_time + ed_time) / 2.0
        kf = min(track, key=lambda e: abs(e["ts"] - mid))
        keyframe_ts, keyframe_bbox = kf["ts"], kf["bbox"]

    video = os.path.join(video_dir, str(vid)) if video_dir else str(vid)
    evidence = []
    if keyframe_bbox is not None:
        evidence.append({
            "object": caption,                             # HC-STVG目标是人物, 用描述句做定位文本
            "interval": [st_time, ed_time],
            "keyframe_ts": keyframe_ts,
            "bbox": keyframe_bbox,
            "track": track,
            "pseudo_bbox": False,
        })
    return {
        "video": video,
        "query": caption,
        "object": caption,                                 # 定位/选帧都用这句描述(人物中心)
        "gold": {"text": [caption], "evidence": evidence},
        "meta": {"source": "hcstvg", "fps": round(fps, 3), "wh": [s.get("width"), s.get("height")]},
    }


def _iter_entries(data):
    """HC-STVG 标注可能是 {vid: {...}} 或 [ {...,'vid':..}, ... ]。统一成 (vid, entry)。"""
    if isinstance(data, dict):
        for k, v in data.items():
            if isinstance(v, dict):
                yield k, v
    elif isinstance(data, list):
        for v in data:
            yield str(v.get("vid") or v.get("video") or v.get("vid_name") or ""), v


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in", dest="inp", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--video_dir", default="")
    a = ap.parse_args()
    data = json.load(open(a.inp, encoding="utf-8"))
    n_in, n_out = 0, 0
    with open(a.out, "w", encoding="utf-8") as f:
        for vid, s in _iter_entries(data):
            n_in += 1
            if not s.get("bbox"):
                continue
            f.write(json.dumps(convert_one(vid, s, a.video_dir), ensure_ascii=False) + "\n")
            n_out += 1
    print(f"[done] {n_in} -> {n_out} 条 VKIG  ({a.out})")


def selftest():
    s = {"caption": "The woman in red clothes turns in a circle.",
         "bbox": [[100, 50, 40, 120], [110, 55, 40, 120], [120, 60, 42, 122], [130, 62, 42, 122], [140, 64, 44, 124]],
         "st_time": 2.0, "ed_time": 4.0, "st_frame": 61, "img_num": 500, "width": 640, "height": 360}
    # fps=500/20=25; st_frame=61 -> 首框时间=(61-1)/25=2.4s; 逐帧+1/25=0.04s
    out = convert_one("v001.mp4", s, video_dir="/data")
    assert out["video"] == "/data/v001.mp4"
    assert out["query"].startswith("The woman")
    ev = out["gold"]["evidence"][0]
    assert ev["interval"] == [2.0, 4.0]
    assert len(ev["track"]) == 5
    # 首框 [100,50,40,120] -> xyxy [100,50,140,170]
    assert ev["track"][0]["bbox"] == [100.0, 50.0, 140.0, 170.0], ev["track"][0]["bbox"]
    assert abs(ev["track"][0]["ts"] - 2.4) < 1e-6, ev["track"][0]["ts"]
    # 中点 mid=3.0s; track时间=[2.4,2.44,2.48,2.52,2.56] -> 最近3.0的是最后一个(2.56)
    assert abs(ev["keyframe_ts"] - 2.56) < 1e-6, ev["keyframe_ts"]
    assert out["meta"]["fps"] == 25.0
    # dict 与 list 两种入口
    assert next(_iter_entries({"a": {"caption": "x"}}))[0] == "a"
    assert next(_iter_entries([{"vid": "b", "caption": "y"}]))[0] == "b"
    print("✅ hcstvg_to_vkig selftest 通过: xywh->xyxy / fps=img_num/20 / 帧号->时间 / 中点关键帧 / dict&list入口")


if __name__ == "__main__":
    import sys
    selftest() if "--selftest" in sys.argv else main()
