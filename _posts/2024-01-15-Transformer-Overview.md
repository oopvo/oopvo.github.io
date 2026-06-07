---
layout: post
title: Transformer 架构详解：从 Attention 到 Multi-Head
categories: deep-learning
tags: [Transformer, Attention, NLP]
---

**{% include gloss.html term="Transformer" %}** 架构自 2017 年由 Google 提出以来，已经成为自然语言处理乃至整个深度学习领域最重要的基础架构之一。从 BERT、GPT 到如今的 DeepSeek，几乎所有主流大模型都建立在 {% include gloss.html term="Transformer" %} 的基础之上。

本文将从最基础的 {% include gloss.html term="Attention" %} 机制开始，逐步深入到 {% include gloss.html term="Multi-Head Attention" %}、{% include gloss.html term="Positional Encoding" %} 等核心组件。

---

## 为什么需要 Attention

在 {% include gloss.html term="Transformer" %} 出现之前，主流序列模型是 RNN/LSTM。它们的问题：

```
RNN 处理长序列:
  "我" → "爱" → "深" → "度" → "学" → "习"
   ↑信息逐  步衰  减  →  到  →  这  →  里

  → 长距离依赖难以捕捉
  → 只能顺序计算（无法并行）

Attention:
  "我"─────┐
  "爱"─────┤
  "深"─────┤──→ 直接关注所有位置
  "度"─────┤
  "学"─────┤
  "习"─────┘

  → 一步到位建立直接连接
  → 所有位置可并行计算
```

{% include gloss.html term="Attention" %} 的核心思想：让模型在处理每个位置时，能够动态关注输入序列中的**所有其他位置**，从而一步捕捉长距离依赖。

---

## Scaled Dot-Product Attention

这是 {% include gloss.html term="Attention" %} 最基本的实现形式，也是 {% include gloss.html term="Transformer" %} 所有注意力组件的基础。

### 计算步骤

```
输入: Q（查询）, K（键）, V（值）

Step 1:  分数 = Q × K^T           计算每个 query 和所有 key 的相似度
Step 2:  分数 /= √d_k              缩放，防止 softmax 梯度过小
Step 3:  权重 = softmax(分数)       归一化为概率分布
Step 4:  输出 = 权重 × V          加权聚合值

最终公式:

$$ \text{Attention}(Q, K, V) = \text{softmax}\left(\frac{QK^T}{\sqrt{d_k}}\right) V $$

```

### 直观理解

- **Q（Query）**：当前位置的"问题"——我要找什么？
- **K（Key）**：所有位置的"标签"——我有什么？
- **V（Value）**：所有位置的"内容"——我给出什么？

通过 Q 和 K 的匹配找到最相关的信息，然后用 V 把这些信息提取出来。

### 代码实现

```python
import torch
import torch.nn.functional as F

def scaled_dot_product_attention(Q, K, V):
    # Q, K, V: (batch, heads, seq_len, d_k)
    d_k = Q.size(-1)
    scores = torch.matmul(Q, K.transpose(-2, -1))
    scores = scores / (d_k ** 0.5)
    weights = F.softmax(scores, dim=-1)
    return torch.matmul(weights, V), weights
```

---

## Multi-Head Attention

{% include gloss.html term="Multi-Head Attention" %} 不满足于单一的注意力计算，而是将 Q、K、V 分别投影到多个子空间（头），在每个子空间中**并行执行上述注意力计算**。

### 为什么要多"头"

单一注意力可能只关注到一种模式（如语法关系），而 {% include gloss.html term="Multi-Head Attention" %} 让模型从**多个角度**同时关注信息：

```
头 1:      "猫" → "追" → "老鼠"
          关注主谓宾关系

头 2:      "猫" ←→ "它"
          关注指代关系

头 3:      "猫" ··· "老鼠"
          关注语义关联

... 每个头学习不同类型的关系
```

### 计算流程

```
输入 X
  │
  ├──→ 投影到 头1: Q₁ = XW_Q₁, K₁ = XW_K₁, V₁ = XW_V₁ → Attention₁
  ├──→ 投影到 头2: Q₂ = XW_Q₂, K₂ = XW_K₂, V₂ = XW_V₂ → Attention₂
  ├──→ ...
  │
  └──→ 拼接所有头的输出 → 线性变换 → 最终输出

$$
\text{MultiHead}(Q, K, V) = \text{Concat}(\text{head}_1, \dots, \text{head}_h) W_O
$$

```
MultiHead(Q,K,V) = Concat(head₁,...,head_h) W_O
```

---

## Positional Encoding

{% include gloss.html term="Attention" %} 机制有一个固有问题：它本身是**置换不变的**（permutation invariant）。如果不加位置信息，"A→B" 和 "B→A" 在 Attention 看来是一样的。

{% include gloss.html term="Positional Encoding" %} 通过为每个位置注入唯一的位置信号来解决这个问题。

### 正弦波编码

{% include gloss.html term="Transformer" %} 原论文使用不同频率的正弦/余弦函数：

$$
\begin{aligned}
PE(pos, 2i)   &= \sin\left(\frac{pos}{10000^{2i/d_{\text{model}}}}\right) \\
PE(pos, 2i+1) &= \cos\left(\frac{pos}{10000^{2i/d_{\text{model}}}}\right)
\end{aligned}
$$

- **pos**: 位置索引（0, 1, 2, ...）
- **i**: 维度索引（0, 1, 2, ..., d_model/2）
- 不同维度使用不同频率 → 编码向量包含多尺度位置信息

### RoPE（旋转位置编码）

{% include gloss.html term="Transformer" %} 原始的位置编码是绝对位置编码，而现代模型（如 LLaMA、DeepSeek）多使用 **RoPE**（旋转位置编码），它通过旋转矩阵将位置信息编码到 Q 和 K 中，天然支持**相对位置**感知。

{% include gloss.html term="DeepSeek-V3" %} 的 {% include gloss.html term="MLA" %} 架构就是使用 RoPE 来实现高效的长上下文推理。

---

## Transformer 整体架构

将以上组件组合起来：

```
输出
  ▲
  │
┌───┴──────────────┐
│  层归一化          │
│  前馈网络 (FFN)    │
│  残差连接          │
├───────────────────┤
│  层归一化          │
│  Multi-Head       │
│  Attention         │
│  残差连接          │
├───────────────────┤
│  Positional       │
│  Encoding          │
├───────────────────┤
│  输入嵌入           │
└───────────────────┘
```

每一层都使用了 **残差连接**（类似 {% include gloss.html term="ResNet" %} 中的 {% include gloss.html term="Shortcut Connections" %}），保证梯度可以顺畅回传。

---

## 对学习者的启示

1. **{% include gloss.html term="Attention" %}** 的核心是 QKV 三元的匹配与聚合——理解了这个思想，就理解了 {% include gloss.html term="Transformer" %}
2. **{% include gloss.html term="Multi-Head Attention" %}** 的多角度思路可以推广到其他架构设计中
3. **{% include gloss.html term="Positional Encoding" %}** 提醒我们：模型需要明确的"位置感"
4. {% include gloss.html term="Transformer" %} 的并行计算特性是其取代 RNN 的核心优势

---

## 参考文献

- [Attention Is All You Need, NeurIPS 2017](https://arxiv.org/abs/1706.03762)
- [The Annotated Transformer](http://nlp.seas.harvard.edu/2018/04/03/attention.html)
- [RoFormer: Enhanced Transformer with Rotary Position Embedding](https://arxiv.org/abs/2104.09864)
