# merge_and_eval.py  (可选)
# 1) 合并 LoRA 到基座导出完整权重(swift export --merge_lora);
# 2) 在验证集 JSONL 上批量推理, 用 字符串规范化后精确匹配 算准确率(MUSIC-AVQA 答案是短词/数字)。
# ⚠️ 合并/导出命令以官方为准:
#    https://swift.readthedocs.io/en/latest/BestPractices/Qwen3-VL-Best-Practice.html
#    命令行参数全表: https://swift.readthedocs.io/en/latest/Instruction/Command-line-parameters.html
import os, json, argparse, subprocess, re

os.environ.setdefault("FPS", "2.0")
os.environ.setdefault("FPS_MAX_FRAMES", "16")
os.environ.setdefault("VIDEO_MAX_PIXELS", "50176")
os.environ.setdefault("USE_MODELSCOPE_HUB", "1")


def merge_lora(adapters, out_dir):
    """调用 swift export 把 LoRA 合并进基座, 产出可独立加载的完整模型。"""
    cmd = [
        "swift", "export",
        "--adapters", adapters,
        "--merge_lora", "true",
        "--output_dir", out_dir,
    ]
    print("RUN:", " ".join(cmd))
    subprocess.run(cmd, check=True)
    # 合并后的权重通常在 out_dir 下的 *-merged 目录, 具体名以 swift 输出日志为准
    return out_dir


def norm(s):
    """答案规范化: 小写 + 去标点空白, 便于精确匹配(MUSIC-AVQA 答案如 'two'/'yes'/'piano')。"""
    return re.sub(r"[^a-z0-9]", "", str(s).strip().lower())


def evaluate(model_path, val_jsonl, max_new_tokens=32, limit=None):
    from swift.llm import PtEngine, RequestConfig, InferRequest
    engine = PtEngine.from_pretrained(model_path)

    correct, total = 0, 0
    with open(val_jsonl, "r", encoding="utf-8") as f:
        for i, line in enumerate(f):
            if limit and i >= limit:
                break
            ex = json.loads(line)
            gt = ex["messages"][-1]["content"]              # assistant 答案 = ground truth
            user_msg = ex["messages"][0]                    # 只喂 user 那条
            req = InferRequest(messages=[user_msg], videos=ex["videos"])
            resp = engine.infer([req], RequestConfig(max_tokens=max_new_tokens, temperature=0.0))
            pred = resp[0].choices[0].message.content
            total += 1
            if norm(pred) == norm(gt):
                correct += 1
            if total % 20 == 0:
                print(f"  ...{total} done, acc={correct/total:.3f}")
    acc = correct / total if total else 0.0
    print(f"[eval] {correct}/{total} = {acc:.4f}")
    return acc


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--adapters", required=True)
    ap.add_argument("--val", required=True, help="swift_val.jsonl")
    ap.add_argument("--merged_dir", default="/root/autodl-tmp/output/merged")
    ap.add_argument("--skip_merge", action="store_true",
                    help="不合并, 直接用 adapters 评测(PtEngine 也能直接吃 adapter)")
    ap.add_argument("--limit", type=int, default=None, help="只评前 N 条(快速验证)")
    args = ap.parse_args()

    model_path = args.adapters if args.skip_merge else merge_lora(args.adapters, args.merged_dir)
    evaluate(model_path, args.val, limit=args.limit)


if __name__ == "__main__":
    main()
