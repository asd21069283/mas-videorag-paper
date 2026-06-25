# infer.py
# 加载训练好的 LoRA adapter, 对一个视频+问题做推理并打印答案。
# 用 ms-swift 的 Python 推理引擎(PtEngine), 对应 CLI 的 `swift infer --adapters ...`。
# ⚠️ ms-swift 的 Python API 在不同大版本间有微调, 以官方为准:
#    https://swift.readthedocs.io/en/latest/BestPractices/Qwen3-VL-Best-Practice.html
import os
import argparse

# 同样的省显存抽帧设置(推理也要, 否则长视频会爆/慢)
os.environ.setdefault("FPS", "2.0")
os.environ.setdefault("FPS_MAX_FRAMES", "16")
os.environ.setdefault("VIDEO_MAX_PIXELS", "50176")
os.environ.setdefault("USE_MODELSCOPE_HUB", "1")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--adapters", required=True, help="LoRA checkpoint 目录")
    ap.add_argument("--video", required=True, help="测试视频路径")
    ap.add_argument("--question", required=True, help="问题文本")
    ap.add_argument("--max_new_tokens", type=int, default=128)
    args = ap.parse_args()

    from swift.llm import PtEngine, RequestConfig, InferRequest

    # adapters 传 LoRA 目录, 引擎会自动读取里面记录的 base model 并挂上 adapter
    engine = PtEngine.from_pretrained(args.adapters)

    infer_request = InferRequest(
        messages=[{"role": "user", "content": f"<video>{args.question}"}],
        videos=[os.path.abspath(args.video)],
    )
    resp = engine.infer(
        [infer_request],
        RequestConfig(max_tokens=args.max_new_tokens, temperature=0.0),
    )
    print("Q:", args.question)
    print("A:", resp[0].choices[0].message.content)


if __name__ == "__main__":
    main()

# ---- 等价 CLI(不想写 Python 时直接用, 最稳妥) ----
# FPS_MAX_FRAMES=16 VIDEO_MAX_PIXELS=50176 \
# swift infer \
#     --adapters /root/autodl-tmp/output/qwen3vl-musicavqa/vX-xxxx/checkpoint-xxx \
#     --stream true \
#     --max_new_tokens 128 \
#     --val_dataset /path/to/one_sample.jsonl   # 或交互式手动贴 <video> 路径+问题
