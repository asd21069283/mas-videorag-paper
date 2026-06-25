# prepare_musicavqa.py
# 把 MUSIC-AVQA 官方标注(avqa-*.json)转成 ms-swift 训练需要的 messages+videos JSONL。
# ms-swift 视频样本格式(已核对官方 Best Practice):
#   {"messages": [{"role":"user","content":"<video>问题"},
#                 {"role":"assistant","content":"答案"}],
#    "videos": ["/abs/path/xxx.mp4"]}
# 官方多模态格式说明: https://swift.readthedocs.io/en/latest/BestPractices/Qwen3-VL-Best-Practice.html
import argparse, json, os, ast


def load_anns(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def fill_template(question_content, templ_values_raw):
    """MUSIC-AVQA 的问题里有 <Object>/<LR> 等占位符, 真实取值在 templ_values。
    templ_values 在原始 JSON 里是【被转义的字符串】, 例如 "[]" 或 "[\"piano\"]",
    需要先解析成 list, 再按出现顺序把每个占位符(<...>)替换掉。"""
    try:
        values = ast.literal_eval(templ_values_raw) if isinstance(templ_values_raw, str) else templ_values_raw
    except (ValueError, SyntaxError):
        values = []
    if not values:
        return question_content
    q = question_content
    # 占位符形如 <xxx>, 依次用 templ_values 里的值替换
    import re
    placeholders = re.findall(r"<[^>]+>", q)
    for ph, val in zip(placeholders, values):
        q = q.replace(ph, str(val), 1)
    return q


def find_video(video_dir, video_id):
    """MUSIC-AVQA 视频文件名一般就是 video_id, 但扩展名可能是 .mp4 等, 容错查找。"""
    for ext in (".mp4", ".mkv", ".avi", ".webm", ".mov"):
        p = os.path.join(video_dir, video_id + ext)
        if os.path.exists(p):
            return p
    # 兜底: 直接拼 .mp4(后续训练若文件缺失会报错, 便于发现)
    return os.path.join(video_dir, video_id + ".mp4")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ann", required=True, help="MUSIC-AVQA 标注 json, 如 avqa-train.json")
    ap.add_argument("--video_dir", required=True, help="9288 个视频所在目录")
    ap.add_argument("--out", required=True, help="输出 ms-swift JSONL")
    ap.add_argument("--require_video_exists", action="store_true",
                    help="开启后跳过磁盘上找不到的视频(子集跑通流程时很有用)")
    args = ap.parse_args()

    anns = load_anns(args.ann)
    n_in, n_out, n_skip = 0, 0, 0
    with open(args.out, "w", encoding="utf-8") as fout:
        for a in anns:
            n_in += 1
            # 已删除的问题不要
            if a.get("question_deleted", 0):
                continue
            vid = str(a["video_id"])
            video_path = find_video(args.video_dir, vid)
            if args.require_video_exists and not os.path.exists(video_path):
                n_skip += 1
                continue

            question = fill_template(a["question_content"], a.get("templ_values", "[]"))
            answer = str(a["anser"])  # ⚠️ 官方字段就拼成 anser, 不是 answer

            sample = {
                "messages": [
                    # <video> 占位符告诉模型这里插入视频, 实际文件由 videos 字段给
                    {"role": "user", "content": f"<video>{question}"},
                    {"role": "assistant", "content": answer},
                ],
                "videos": [os.path.abspath(video_path)],
            }
            fout.write(json.dumps(sample, ensure_ascii=False) + "\n")
            n_out += 1

    print(f"[done] 读入 {n_in} 条 -> 写出 {n_out} 条, 跳过(无视频) {n_skip} 条")
    print(f"输出: {args.out}")


if __name__ == "__main__":
    main()
