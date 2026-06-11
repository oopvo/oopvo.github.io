---
layout: post
title: Llama 系列技术报告精读（未完善）
date: 2026-04-06
categories: paper-reading
tags: [Llama, GQA, SwiGLU, 开源模型, 大模型]
date: 2026-06-06
---

## 概述

**{% include gloss.html term="Llama" %}**（Large Language Model Meta AI）是 Meta 发布的开源大语言模型系列，从 Llama 1 到 Llama 4，每一代都推动了开源模型的边界。{% include gloss.html term="Llama" %} 系列引入了多项关键技术创新，包括 {% include gloss.html term="GQA" %}、{% include gloss.html term="SwiGLU" %} 和 {% include gloss.html term="RMSNorm" %}。

> 论文：[Llama 1](https://arxiv.org/abs/2302.13971) | [Llama 2](https://arxiv.org/abs/2307.09288) | [Llama 3](https://arxiv.org/abs/2407.21783)

---

## 发展简史

```
Llama 1 (2023.02)      — 7B/13B/33B/65B，开源先驱
    │
Llama 2 (2023.07)      — 7B/13B/70B，引入 GQA，开源商用
    │
Llama 3 (2024.04)      — 8B/70B/405B，史上最大开源密集模型
    │
Llama 4 (2025.04)      — MoE 架构，首次引入混合专家
```

---

## 一、{% include gloss.html term="GQA" %}：Grouped-Query Attention

{% include gloss.html term="Llama" %} 2 在 70B 版本中首次引入 {% include gloss.html term="GQA" %}，这是 {% include gloss.html term="Llama" %} 系列在注意力机制上的关键创新。

### MHA → GQA → MQA 的演进

```
标准 MHA（Llama 1）:
  Q: 64 头    K: 64 头    V: 64 头
  KV 缓存 = 64 × head_dim × layers → 推理时内存爆炸

GQA（Llama 2 70B / Llama 3）:
  Q: 64 头    K: 8 组     V: 8 组
  KV 缓存 = 8 × head_dim × layers → 减少 87.5%

  Q 头分 8 组，每组 8 个头共享一个 K/V
  → 质量接近 MHA，缓存接近 MQA
```

### 为什么 GQA 有效

在推理时，{% include gloss.html term="GQA" %} 的核心优势是**减少 KV 缓存的数据搬运量**。自回归生成中，KV 缓存需要从 HBM 读取到 SRAM，访存带宽通常是瓶颈。KV 缓存减少 87.5% 意味着：

- 同样硬件可以支持**更长的上下文**
- 同样上下文可以**提高批量大小**
- 推理延迟显著降低

> 注：DeepSeek 的 {% include gloss.html term="MLA" %} 更进一步，将 KV 压缩到潜在空间，缓存仅相当于 MHA 的 1.4%。

---

## 二、{% include gloss.html term="SwiGLU" %}：门控激活函数

{% include gloss.html term="Llama" %} 全系列使用 {% include gloss.html term="SwiGLU" %} 作为激活函数，替代了早期模型常用的 ReLU 或 GELU。

### 公式对比

```
ReLU:      output = max(0, xW)               简单但表达力有限
GELU:      output = x · Φ(x)                 平滑，略有提升
SwiGLU:    output = Swish(xW) ⊗ (xV)         门控机制，最强
```

{% include gloss.html term="SwiGLU" %} 通过引入门控机制（第二个投影矩阵 V），让网络可以**动态选择**哪些信息通过，类似 LSTM 中的门控思想。

### 参数代价

{% include gloss.html term="SwiGLU" %} 的代价是额外增加了一个投影矩阵 V，使得 FFN 层的参数量增加约 1/3。作为回报，在相同计算量下，{% include gloss.html term="SwiGLU" %} 比 ReLU 提升约 **0.5-1.0%** 的模型质量。

---

## 三、{% include gloss.html term="RMSNorm" %}：简化的层归一化

{% include gloss.html term="Llama" %} 系列使用 {% include gloss.html term="RMSNorm" %} 替代标准的 LayerNorm。

### 为什么可以简化

LayerNorm 做了两件事：**中心化**（减均值）和 **缩放**（除方差）。{% include gloss.html term="RMSNorm" %} 发现，在 Transformer 中，中心化步骤不是必需的——只做缩放就够。

```
LayerNorm:  y = (x - μ) / σ × γ + β     ← 两步都做
RMSNorm:    y = x / RMS(x) × γ          ← 只做缩放

其中 RMS(x) = √(1/n × Σx²)
```

### 实际收益

- 计算量减少约 **10-15%**
- 模型质量**无损**
- 被 Llama、Qwen、Mistral 等主流模型广泛采用

---

## 四、Llama 3 405B：开源密集模型的巅峰

{% include gloss.html term="Llama" %} 3 405B 是当时最大的开源密集模型（非 MoE）。

### 模型配置

| 参数 | 数值 |
|------|------|
| 层数 | **126 层** |
| 隐藏维度 | **16,384** |
| FFN 维度 | 53,248 |
| 注意力头 | 128 头，{% include gloss.html term="GQA" %}（8 K/V 头）|
| 词表大小 | 128,000 tokens |
| 上下文 | 128K（RoPE, θ=500,000）|
| 训练数据 | **15.6T tokens** |
| GPU | 30.8M H100 GPU 小时 |

### Scaling Law：为什么是 405B？

Meta 通过 **Chinchilla Scaling Law** 确定了 405B 是最优参数量：

```
训练一个 Transformer 的计算量 ≈ 6 × 参数量 × 数据量
                 ↓
给定 3.8×10²⁵ FLOPs 的预算：
  更大的模型 + 更少数据 = 欠拟合 ❌
  更小的模型 + 更多数据 = 过拟合 ❌
  405B + 15.6T tokens = 计算最优 ✅（performance per FLOP 最高）
```

他们还发现了**能力突现的 S 曲线**：

```
基准准确率
   1.0 |                          🚀
       |                       ↗
   0.5 |                 ↗
       |           ↗
   0.0 |  —————————
       +——————————————————→ 训练计算量
         能力突然涌现！到达某个阈值后迅速提升
```

### 4D 并行：如何训练 405B

405B 参数在 FP16 下需要 810GB 显存，远超单块 H100 的 80GB。Llama 3 使用 **4D 并行**来分布在 16,384 块 GPU 上：

```
4D 并行策略（4 个维度同时切分）：

① 数据并行（DP）= 切分 batch
   每块 GPU 有完整模型，处理不同数据 → 梯度同步
   
② 张量并行（TP）= 切分层内参数
   一个 Transformer 层的矩阵被切到多块 GPU 上 ← 最常用

③ 流水线并行（PP）= 切分层间
   第 1-30 层在 GPU A，第 31-60 层在 GPU B...

④ 上下文并行（CP）= 切分序列长度
   长上下文时把序列切成多段，分别在不同 GPU 上处理

典型配置（512 节点 × 4096 GPU）:
  TP=8, PP=8, CP=2, DP=4 → 8×8×2×4 = 512 组 × 8 GPU = 4096 ✅
```

### 训练三阶段

```
Phase 1 — 初始预训练:
  上下文: 8K tokens
  batch size: 逐步增加到 16M tokens
  目标: 学习基础知识

Phase 2 — 长上下文预训练:
  上下文: 8K → 128K 逐步扩展
  额外: ~800B tokens
  目标: 支持长序列推理

Phase 3 — 退火（Annealing）:
  最后 40M tokens
  学习率线性衰减到 0
  高质量数据上采样（数学、代码、逻辑）
  Polyak 平均化检查点
```

### 后训练：迭代 SFT + DPO

{% include gloss.html term="Llama" %} 3 的后训练不采用 PPO，而是使用 **迭代 SFT + DPO**：

```
第 1 轮: 
  模型生成多个回答 → RM 评分 → 选最好的 → SFT 微调 → DPO 对齐
  
第 2 轮:
  更强的模型 → 生成更高质量的回答 → SFT → DPO
  
...
每轮生成的合成数据质量都在提升（正反馈循环 ✨）
```

### 性能对比

| 基准测试 | Llama 3 405B | GPT-4 | Claude 3.5 |
|---------|-------------|-------|-----------|
| MMLU | **87.8** | 86.4 | — |
| HumanEval | **89.0** | 87.0 | — |
| GSM-8K | **96.8** | 95.3 | — |
| MATH | **72.4** | — | — |

### 与 DeepSeek-V3 的对比

| 维度 | Llama 3 405B | DeepSeek-V3 |
|------|-------------|------------|
| 架构 | **密集**（Dense） | **MoE** |
| 总参数量 | 405B | **671B** |
| 激活参数 | 405B（100%） | **37B（5.5%）** |
| 训练 GPU 时 | 30.8M | **2.788M（11× 效率差距）** |

{% include gloss.html term="Llama" %} 3 选择密集架构以保证训练稳定性和推理可预测性，而 {% include gloss.html term="MoE" %} 方案虽然效率更高，但工程复杂度也更高。

---

## 五、Llama 4：MoE 转型

2025 年 4 月，{% include gloss.html term="Llama" %} 4 发布，首次采用 {% include gloss.html term="MoE" %} 架构：

- Llama 4 Scout：109B 总参数，17B 激活（适合单 GPU 部署）
- Llama 4 Maverick：402B 总参数，~40B 激活
- 训练数据量提升至 30T+ tokens
- 上下文窗口扩展至 10M tokens（基于密集注意力）

---

## 对开发者的启示

1. **{% include gloss.html term="GQA" %}** 是性价比最高的注意力优化——实现简单，收益显著
2. **{% include gloss.html term="SwiGLU" %}** 虽然增加了参数，但质量收益值得
3. **{% include gloss.html term="RMSNorm" %}** 证明有时候「简化」比「优化」更好
4. 密集模型（{% include gloss.html term="Llama" %} 3）和 {% include gloss.html term="MoE" %}（Llama 4）各有适用场景，不是越"先进"越好
5. Llama 的开源策略推动了整个生态的发展

---

## 参考文献

- [Llama 1: Open and Efficient Foundation Language Models](https://arxiv.org/abs/2302.13971)
- [Llama 2: Open Foundation and Fine-Tuned Chat Models](https://arxiv.org/abs/2307.09288)
- [The Llama 3 Herd of Models](https://arxiv.org/abs/2407.21783)
- [GQA: Training Generalized Multi-Query Transformer Models](https://arxiv.org/abs/2305.13245)
