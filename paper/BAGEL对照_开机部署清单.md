# BAGEL 强模型对照 — 开机即用部署清单(2026-07-22 调研)

**目的**：用一个论文点名的强统一模型(BAGEL-7B-MoT)替 Janus 做 monolith 对照,堵"故意挑弱模型"的 straw-man 质疑。BAGEL 的关键优势:**原生支持图像条件生成(🖌️ Image Edit)**——能直接"看着关键帧生成",若在此最有利设置下仍无定位输出/仍不忠实,论证顶格。

## 可行性判断:**艰难可行,环境坑深,需我在线盯**

| 维度 | 情况 | 应对 |
|---|---|---|
| 显存 | 全精度装不进 24G;**FP8 量化版**(`meimeilook/BAGEL-7B-MoT-FP8`)专为 24G 配,+CPU offload 可跑 | 用 FP8 版,`NUM_ADDITIONAL_LLM_LAYERS_TO_GPU=5` |
| 速度 | offload 下约 1.5× 慢 | 20 条可接受,预留时间 |
| **盘** | FP8 权重 + base repo 估 15-20GB;当前盘只剩 ~20G(Janus 占 11G) | **先删 Janus_pro_7b 释放 11G**,再下 BAGEL |
| **依赖(最大坑)** | flash_attn 特定 wheel + PyTorch 2.5.1/CUDA12.4/Py3.10 **严格版本匹配** | **独立 conda env**(勿污染现有 Qwen3-VL 环境);flash_attn 装官方指定 wheel |

## 开机执行顺序(我在线盯,预留 2-3 小时含搭环境)
1. 删 `/root/autodl-tmp/janus_pro_7b` 腾盘(Janus 结果已拉回本机,可删)
2. 新建 conda env(Py3.10)+ 装 PyTorch 2.5.1+cu124 + 指定 flash_attn wheel + requirements(注释掉旧 flash_attn 行)
3. hf-mirror 下 BAGEL-7B-MoT base(排除原始 ema.safetensors)+ FP8 权重
4. **冒烟 1 条**(图像条件模式:喂关键帧+问题→生成)——过了再全量
5. 20 条同协议(用 §6.8 同一批 vid,可直接同 vid 对比)→ 拉回标注

## 决策点(开机前定)
- **A. 直接上 BAGEL-FP8**:论证最强,但环境搭建有搭不起来的风险(flash_attn 版本地狱),可能烧 1 小时环境时间
- **B. 先扩实例盘/显存**:BAGEL 全精度就好装,环境简单,但要花钱扩配置
- **C. 退而求其次换 Show-o2**:论文也点名,可能比 BAGEL 好装(待查),但没有 BAGEL 的图像条件优势

**倾向 A**,但接受"环境搭不起来就当场转 C"的预案。冒烟不过不硬耗。

## 来源
- FP8 版配置: `meimeilook/BAGEL-7B-MoT-FP8`(24GiB 默认、offload、flash_attn wheel 严格匹配)
- base: `ByteDance-Seed/BAGEL-7B-MoT`
