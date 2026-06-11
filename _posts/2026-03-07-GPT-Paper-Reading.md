---
layout: post
title: GPT 系列精读
date: 2026-03-07
categories: paper-reading
tags: [GPT, Autoregressive, Transformer, 大模型, 入门]
date: 2026-06-07
---

> {% include gloss.html term="GPT" %} 系列彻底改变了人工智能的格局。从 2018 年 GPT-1 的 1.17 亿参数到 GPT-4 的多模态能力，这篇带你完整走一遍 GPT 的进化之路。

---

## 一、GPT 是什么？

{% include gloss.html term="GPT" %}（Generative Pre-trained Transformer）是 OpenAI 开发的 {% include gloss.html term="Autoregressive" %} 语言模型系列。核心思想很简单：

> **在互联网级文本上预训练 → 在各种任务上微调（或零样本直接使用）**

与 {% include gloss.html term="BERT" %} 的编码器架构不同，{% include gloss.html term="GPT" %} 使用**解码器-only**架构——只保留了 Transformer 的解码器部分，去掉编码器-解码器交叉注意力。

---

## 二、{% include gloss.html term="Autoregressive" %} 生成方式

{% include gloss.html term="GPT" %} 使用**自回归**方式生成文本：逐一预测下一个 token。

### 生成过程

```
输入: "I love"
         ↓
模型计算: P(next | "I love")
         ↓
预测: "learning"（概率最高）
         ↓
输入: "I love learning"
         ↓
模型计算: P(next | "I love learning")
         ↓
预测: "."
...
```

### 因果掩码（Causal Masking）

自回归的关键是**因果掩码**——每个位置只能关注它自己及之前的位置：

```
注意力矩阵（「我 爱 深 度 学 习」）:

    我  爱  深  度  学  习
我  [●,  ✗,  ✗,  ✗,  ✗,  ✗]  ← 只能看自己
爱  [●,  ●,  ✗,  ✗,  ✗,  ✗]  ← 能看"我"和"爱"
深  [●,  ●,  ●,  ✗,  ✗,  ✗]
度  [●,  ●,  ●,  ●,  ✗,  ✗]
学  [●,  ●,  ●,  ●,  ●,  ✗]
习  [●,  ●,  ●,  ●,  ●,  ●]  ← 最后一个能看到所有

实现方式：未来位置分数设为 -∞，Softmax 后为 0
```

---

## 三、GPT 家族发展

### GPT-1（2018.06）：证明可行性

- **1.17 亿参数**，12 层 Transformer 解码器
- 在 BooksCorpus 上预训练
- **核心贡献**：首次证明 Transformer 解码器可以在大规模无标注数据上预训练，再通过微调迁移到下游任务

### GPT-2（2019.02）：零样本的震撼

- **15 亿参数**，48 层
- 在 WebText（800 万网页）上预训练
- **核心贡献**：展示零样本迁移能力——不需要微调，给几个示例就能完成任务

```
GPT-2 的零样本能力：
  输入: "翻译成中文：I love learning →"
  输出: "我爱学习"
  
  没有专门训练过翻译，但通过大量的互联网文本学会了
  → 这就是「上下文学习（In-Context Learning）」的雏形
```

### GPT-3（2020.05）：大模型时代的开端

- **1750 亿参数**，96 层
- 在 Common Crawl + WebText2 + Books + Wikipedia 上训练
- 训练成本：约 $12M
- **核心贡献**：**Scaling Law**——模型越大，能力越强，涌现出小模型没有的能力

```
GPT-3 的涌现能力：
  小模型做不到 → 到一定规模突然能做

  • 上下文学习：给 1-2 个示例就能理解任务
  • 代码生成：写 Python、JavaScript
  • 算术推理：多位加减法
  • 翻译、问答、创意写作...
```

### GPT-4（2023.03）：多模态飞跃

- 参数量未公开（估计 1.5T+，{% include gloss.html term="MoE" %} 架构）
- 多模态：可输入图像
- **核心贡献**：推理能力大幅提升，在各种专业考试中表现优异

```
GPT-4 的考试成绩：
  Uniform Bar Exam:     ~90%  percentile  ← 超过大部分人类律师
  SAT 阅读/写作:        710/800
  AP 生物学:            5/5（满分）
  编程竞赛 (Codeforces): 超过 50% 参赛者
```

---

## 四、GPT 与 BERT：两种范式对比

| 维度 | GPT | BERT |
|------|-----|------|
| 架构 | **解码器-only** | **编码器-only** |
| 注意力 | 单向（因果掩码） | **双向** |
| 训练目标 | 自回归（预测下一个词）| {% include gloss.html term="MLM" %}（掩码预测）|
| 适合任务 | **生成**（对话、创作）| **理解**（分类、抽取）|
| 发展方向 | 模型规模 Scaling | 模型深度 + 双向理解 |
| 影响力 | GPT-3 开创大模型时代 | BERT 开启预训练+微调范式 |
| 后继 | GPT-4、Llama、DeepSeek | RoBERTa、ALBERT、DistilBERT |

两者都基于 {% include gloss.html term="Transformer" %} 架构，但设计哲学不同。历史证明，**自回归解码器架构最终胜出**，成为现代大模型（GPT-4、Llama、DeepSeek）的标准选择。

---

## 五、GPT 系列的影响

1. **Scaling Law**：GPT-3 证明了模型规模与能力之间的正相关关系
2. **上下文学习**：不需要为每个任务微调，prompt 工程成为新范式
3. **ChatGPT**：GPT-3.5 + RLHF 引发了全球 AI 热潮
4. **GPT-4**：多模态 + 推理能力接近人类专家水平

---

## 参考文献

- [GPT-1: Improving Language Understanding by Generative Pre-Training](https://cdn.openai.com/research-covers/language-unsupervised/language_understanding_paper.pdf)
- [GPT-2: Language Models are Unsupervised Multitask Learners](https://d4mucfpksywv.cloudfront.net/better-language-models/language-models.pdf)
- [GPT-3: Language Models are Few-Shot Learners](https://arxiv.org/abs/2005.14165)
- [GPT-4 Technical Report](https://arxiv.org/abs/2303.08774)
