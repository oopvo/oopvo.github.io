---
layout: post
title: Transformer 架构详解（未完善）
categories: deep-learning
tags: [Transformer, Attention, NLP, 入门]
---

> 如果你只能读一篇关于 Transformer 的文章，希望是这一篇。本文从最基础的问题出发，一步步推导出 Transformer 的每一个核心设计，保证看完能理解为什么它如此强大。

---

## 一、Transformer 要解决什么问题？

在 {% include gloss.html term="Transformer" %} 出现之前（2017 年以前），主流的序列模型是 **RNN（循环神经网络）** 和 **LSTM**。它们有什么问题？

### RNN 的两大痛点

**痛点 1：长距离依赖丢失**

```
RNN 处理句子 "我在北京出生，后来去了上海读书，现在……喜欢这里"

  "我" → "在" → "北京" → "出生" → …… → "上海" → "读书" → …… → "这里"
   ↑信息  一步步  传递   逐层   衰减   到   这里   已经   模糊  不清

  问题："这里" 指的是 "北京" 还是 "上海"？
  → 信息经过太多步，早就模糊了
```

**痛点 2：不能并行计算**

```
RNN 必须按顺序计算：
  第 1 步: "我" → 隐藏状态 h₁
  第 2 步: "在" → 隐藏状态 h₂（依赖 h₁）
  第 3 步: "北京" → 隐藏状态 h₃（依赖 h₂）
  ...
  
  第 N 步必须等前 N-1 步算完 → 不能并行 → 训练慢
```

### Transformer 的核心洞察

**一步到位建立连接**——不再逐个传递信息，而是让每个词直接"看"到所有其他词：

```
传统 RNN:     我 → 在 → 北京 → …… → 喜欢 → 这里
                                 信息逐层衰减↗

Transformer:  我 ──────┐
              在 ──────┤
              北京 ────┤──→ 直接关注所有位置
              ……  ────┤
              喜欢 ────┤
              这里 ────┘
              
              每个词一步就连接到其他所有词！⚡
```

---

## 二、{% include gloss.html term="Attention" %}：注意力机制

### 直观理解

想象你在读这句话："**那个穿红色衣服的女孩跑过去捡起了她的书包**"。

当看到"**她**"时，你会自然地关注到"**女孩**"——因为"她"指的就是"女孩"。这就是注意力：**在处理某个位置时，知道应该关注输入的哪些其他位置**。

### 经典例子："it" 指的是什么？

来看 Jay Alammar 的经典例子：

> **"The animal didn't cross the street because it was too tired"**
> （那只动物没有过马路，因为它太累了）

**问题**：这里的 "it" 指的是 "animal" 还是 "street"？

人类一眼就知道是 "animal"（动物太累了），但机器需要学会这种指代关系。这正是 {% include gloss.html term="Attention" %} 要解决的问题：

```
当模型处理 "it" 这个位置时：

  注意力权重分布（越高 = 越关注）:
  
  The    animal    didn't    cross    the    street    because    it    was    tired
  0.01    0.72     0.01      0.01    0.01    0.08      0.01      —    0.01   0.14
  
  ↑ "it" 把 72% 的注意力放在 "animal" 上
  ↑ 8% 关注 "street"（也有可能是街太累了？）
  ↑ 14% 关注 "tired"（"累了"的主体是 animal）
  
  → 模型正确理解了 "it" → "animal" 的指代关系 ✅
```

这种**注意力可视化**是 Transformer 可解释性的重要工具——我们能看到模型在关注什么。

### 数学形式：QKV 三要素

{% include gloss.html term="Attention" %} 的核心是三个矩阵：**Q（Query）**、**K（Key）**、**V（Value）**。

```python
# 用生活类比理解 QKV
Q = Query   = "你在搜索引擎里输入的问题"
K = Key     = "网页的标题标签"  
V = Value   = "网页的实际内容"
```

**完整的计算分三步：**

**第 1 步：匹配（Q × K）**

每个 Query 和所有 Key 计算相似度得分：

```
  Q₁ → 和 K₁ 的相似度: 0.9  ← "女孩" 和 "女孩" 高度匹配
  Q₁ → 和 K₂ 的相似度: 0.1  ← "女孩" 和 "跑" 不太相关
  Q₁ → 和 K₃ 的相似度: 0.3  ← "女孩" 和 "书包" 有点相关
```

**第 2 步：缩放 + 归一化（Softmax）**

得分除以 √d_k 防止数值过大，然后通过 Softmax 变成概率分布（和为 1）：

```
  原始得分: [0.9, 0.1, 0.3]
  缩放后:  [0.6, 0.07, 0.2]   ← 除以 √d_k，让梯度更稳定
  Softmax: [0.5, 0.2, 0.3]    ← 变成概率，和为 1
```

**第 3 步：加权求和（× V）**

用这些概率加权聚合所有 Value：

```
  输出 = 0.5 × V₁ + 0.2 × V₂ + 0.3 × V₃
  
  最终结果 ≈ 主要包含 V₁（女孩）的信息 + 少量 V₂（跑）+ 一些 V₃（书包）
```

### 完整公式

这就是著名的 **Scaled Dot-Product Attention**：

$$ \text{Attention}(Q, K, V) = \text{softmax}\left(\frac{QK^T}{\sqrt{d_k}}\right) V $$

其中 $d_k$ 是 Key 的维度。除以 $\sqrt{d_k}$ 的原因是：当维度很高时，点积的结果会变得很大，Softmax 的梯度会变得非常小（梯度消失），缩放后解决了这个问题。

### 步骤分解图

```
输入序列: [x₁, x₂, x₃, x₄]

Step 1: 计算 Q、K、V
  x₁ → Q₁, K₁, V₁
  x₂ → Q₂, K₂, V₂
  x₃ → Q₃, K₃, V₃
  x₄ → Q₄, K₄, V₄

Step 2: 计算注意力分数矩阵
       K₁   K₂   K₃   K₄
  Q₁  [0.9, 0.1, 0.3, 0.2] ← x₁ 对各个位置的关注度
  Q₂  [0.2, 0.8, 0.1, 0.4]
  Q₃  [0.3, 0.2, 0.7, 0.1]
  Q₄  [0.1, 0.3, 0.2, 0.9]

Step 3: Softmax 归一化（每行和为 1）

Step 4: 每行 × V 得到输出
  输出₁ = 0.9V₁ × 0.1V₂ × 0.3V₃ × 0.2V₄（加权求和）
```

### 代码实现

```python
import torch
import torch.nn.functional as F

def scaled_dot_product_attention(Q, K, V):
    """
    Q: (batch, heads, seq_len, d_k) — 查询
    K: (batch, heads, seq_len, d_k) — 键
    V: (batch, heads, seq_len, d_v) — 值
    """
    d_k = Q.size(-1)
    
    # Step 1: Q × K^T 计算相似度
    scores = torch.matmul(Q, K.transpose(-2, -1))
    
    # Step 2: 缩放防止梯度消失
    scores = scores / (d_k ** 0.5)
    
    # Step 3: Softmax 归一化
    weights = F.softmax(scores, dim=-1)
    
    # Step 4: 加权求和
    output = torch.matmul(weights, V)
    
    return output, weights  # weights 可以用于可视化
```

---

## 三、{% include gloss.html term="Multi-Head Attention" %}：多头注意力

### 为什么需要"多"头？

单个注意力只能学到**一种**关注模式。但语言很复杂，需要从**多个角度**同时理解：

```
句子: "那只猫追到了老鼠，它跑得很快"

头 1 — 语法关系:    "猫" → "追" → "老鼠"     关注主谓宾
头 2 — 指代关系:    "猫" ← "它"              关注代词指代
头 3 — 语义关联:    "猫" ··· "老鼠"          关注语义关系
头 4 — 位置关系:    "追" → "到" → "了"       关注局部上下文

每个头学到的模式不同 → 拼接起来 → 信息更丰富
```

### 计算流程

```
输入 X
  │
  ├──→ 线形投影到 头 1: Q₁ = XW₁_Q, K₁ = XW₁_K, V₁ = XW₁_V → Attention₁
  ├──→ 线形投影到 头 2: Q₂ = XW₂_Q, K₂ = XW₂_K, V₂ = XW₂_V → Attention₂
  ├──→ 线形投影到 头 3: Q₃ = XW₃_Q, K₃ = XW₃_K, V₃ = XW₃_V → Attention₃
  └──→ 线形投影到 头 4: Q₄ = XW₄_Q, K₄ = XW₄_K, V₄ = XW₄_V → Attention₄
                                              │
                                              ▼
                                 拼接所有头的输出
                                        │
                                        ▼
                                  线形投影 → 最终输出
```

### 完整公式

$$
\text{MultiHead}(Q, K, V) = \text{Concat}(\text{head}_1, \dots, \text{head}_h) W_O
$$

其中每个头为：

$$ \text{head}_i = \text{Attention}(XW_i^Q, XW_i^K, XW_i^V) $$

### 多头 vs 单头的效果

| 特性 | 单头注意力 | 多头注意力（8 头）|
|------|----------|----------------|
| 关注角度 | 1 种 | **8 种不同角度** |
| 参数量 | 较少 | 总参数量相同（每个头维度降低）|
| 表达力 | 有限 | **强得多** |
| 可解释性 | — | 不同头可对应不同语言模式 |

**关键设计细节**：虽然头数增加了，但每个头的维度降低了（如果原来 512 维，8 头就是每个头 64 维），总计算量基本不变。

---

## 四、{% include gloss.html term="Positional Encoding" %}：位置编码

### 为什么需要位置信息？

{% include gloss.html term="Attention" %} 有一个固有问题：**它是置换不变的**（permutation invariant）。

```
不加位置编码时，以下两句话对 Attention 来说完全一样：

  "猫追老鼠"  → 每个词关注其他词的模式
  "老鼠追猫"  → 模式完全相同！因为词袋相同

这显然不对——"猫追老鼠" 和 "老鼠追猫" 意思完全相反！
```

{% include gloss.html term="Positional Encoding" %} 给每个位置一个**唯一的位置信号**，让模型知道词的先后顺序。

### 正弦波位置编码

原论文使用一组正弦/余弦函数，每个维度有不同频率：

$$
\begin{aligned}
PE(pos, 2i)   &= \sin\left(\frac{pos}{10000^{2i/d_{\text{model}}}}\right) \\[4pt]
PE(pos, 2i+1) &= \cos\left(\frac{pos}{10000^{2i/d_{\text{model}}}}\right)
\end{aligned}
$$

**直观理解**：
- $pos$ 是位置（0, 1, 2, ...）
- $i$ 是维度（0, 1, 2, ..., d/2）
- 不同维度用**不同频率**：低维变化快（区分邻近位置），高维变化慢（编码远距离关系）
- 这种设计让模型**既知道绝对位置，也能推断相对距离**

```
维度 0 (高频):  ~ ~ ~ ~ ~ ~ ~ ~   快速变化，区分相邻位置
维度 1:          ~  ~  ~  ~       中等频率
维度 2:           ~   ~   ~       更低频率
...
维度 d (低频):    ~    ~    ~     慢速变化，编码远距离关系
```

### 现代演进：RoPE

{% include gloss.html term="Transformer" %} 原始的位置编码是**绝对位置编码**，直接加到输入上。现代模型（如 {% include gloss.html term="Llama" %}、{% include gloss.html term="DeepSeek-V3" %} 的 {% include gloss.html term="MLA" %}）大多使用 **RoPE（旋转位置编码）**。

RoPE 的核心思想：**不是把位置加到输入上，而是把位置信息编码到 Q 和 K 的旋转矩阵中**。这样 Attention 计算时天然包含相对位置信息。

---

## 五、完整架构：Putting It All Together

### 宏观结构

{% include gloss.html term="Transformer" %} 采用 **Encoder-Decoder**（编码器-解码器）架构：

```
输入序列: ["我", "爱", "学习"]
                │
                ▼
     ┌──────────────────────┐
     │    Encoder（编码器）   │
     │  6 层，每层包含:       │
     │  • Multi-Head Attn   │
     │  • Feed Forward      │
     │  • Add & Norm        │
     └──────────┬───────────┘
                │ 编码后的表示
                ▼
     ┌──────────────────────┐
     │    Decoder（解码器）   │
     │  6 层，每层包含:       │
     │  • Masked Multi-Head  │
     │  • Cross-Attention   │
     │  • Feed Forward      │
     │  • Add & Norm        │
     └──────────┬───────────┘
                │
                ▼
输出序列: ["I", "love", "learning"]
```

### 每一层的内部

```
输入向量 (512 维)
    │
    ▼
┌──────────────────────┐
│  Multi-Head Attention│  ← 捕捉词与词的关系
│  8 头 × 64 维        │
│  残差连接 + 层归一化   │  ← 稳定训练
└──────────────────────┘
    │
    ▼
┌──────────────────────┐
│  Feed Forward Network │  ← 每个位置独立做非线性变换
│  Linear → ReLU → Linear│
│  残差连接 + 层归一化   │
└──────────────────────┘
    │
    ▼
    输出到下一层
```

### 两个关键设计

**1. 残差连接（Residual Connection）**

每条子层输出都加上原始输入：`output = LayerNorm(x + Sublayer(x))`

```
输入 x ──→ [Self-Attention] ──→ ──→ ⊕ ──→ LayerNorm ──→ 输出
                                  ↑
                                  └── ← x（原始输入跳过注意力层）
```

这个设计借鉴了 {% include gloss.html term="ResNet" %} 的 {% include gloss.html term="Shortcut Connections" %}，让梯度可以直接回传，训练更稳定。

**2. 掩码自注意力（Masked Self-Attention）**

Decoder 中的自注意力是**掩码的**——每个位置只能关注它自己及之前的位置，不能"偷看"未来的词。

```
"我 爱 学 习"
                     可关注的
位置 1 ("我"):    [✅, ❌, ❌, ❌]  → 只能看自己
位置 2 ("爱"):    [✅, ✅, ❌, ❌]  → 能看"我"和"爱"
位置 3 ("学"):    [✅, ✅, ✅, ❌]  
位置 4 ("习"):    [✅, ✅, ✅, ✅]

实现方式：把未来位置的分数设为 -∞，Softmax 后为 0
```

---

## 六、为什么 Transformer 能成功？

### RNN vs Transformer 对比

| 维度 | RNN/LSTM | Transformer |
|------|----------|-------------|
| 长距离依赖 | ❌ 逐层传递，容易丢失 | ✅ 一步直达，不衰减 |
| 并行计算 | ❌ 必须顺序计算 | ✅ 所有位置同时计算 |
| 训练速度 | ❌ 慢（串行） | ✅ 快（并行），可 GPU 加速 |
| 长序列 | ❌ 梯度消失/爆炸 | ✅ 残差连接稳定梯度 |
| 可解释性 | ❌ 隐藏状态难解释 | ✅ 注意力权重可可视化 |
| 扩展性 | ❌ 很难做大 | ✅ GPT-4、Llama 3 等千亿参数 |

### 核心贡献总结

1. **完全抛弃 RNN/CNN**——证明纯注意力机制就足够强大
2. **并行计算**——让训练大规模模型成为可能
3. **可扩展**——Transformer 可以堆到上千亿参数
4. **通用性**——不仅 NLP，在 CV、语音、多模态等领域都成为主流

---

## 七、从 Transformer 到现代大模型

{% include gloss.html term="Transformer" %} 之后的发展：

```
Transformer (2017)          ← 基础架构
    │
    ├── BERT (2018)          ← 编码器架构，双向理解
    ├── GPT 系列 (2018-)     ← 解码器架构，文本生成
    ├── T5 (2019)            ← Encoder-Decoder，文本到文本
    │
    ├── Llama 系列 (2023-)   ← 仅解码器，开源优化
    ├── DeepSeek 系列        ← 引入 MLA + MoE
    └── Qwen 系列            ← 多语言优化
```

几乎所有现代大语言模型都基于 {% include gloss.html term="Transformer" %} **解码器架构**，核心创新围绕：
- **注意力优化**：{% include gloss.html term="GQA" %}、{% include gloss.html term="MLA" %}、{% include gloss.html term="DSA" %}
- **架构扩展**：{% include gloss.html term="MoE" %}、{% include gloss.html term="MoDE" %}
- **训练方法**：{% include gloss.html term="SFT" %}、{% include gloss.html term="GRPO" %}、{% include gloss.html term="Self-Play RL" %}

---

## 参考文献

- [Attention Is All You Need, NeurIPS 2017](https://arxiv.org/abs/1706.03762)
- [The Annotated Transformer](http://nlp.seas.harvard.edu/2018/04/03/attention.html)
- [RoFormer: Enhanced Transformer with Rotary Position Embedding](https://arxiv.org/abs/2104.09864)
