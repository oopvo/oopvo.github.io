---
layout: post
title: BERT 论文精读：双向预训练改变 NLP
categories: paper-reading
tags: [BERT, MLM, 预训练, 自然语言理解]
date: 2026-06-07
---

> 2018 年，{% include gloss.html term="BERT" %} 的出现如同 NLP 领域的"ImageNet 时刻"。它在 11 项 NLP 基准上取得 SOTA，开启了「预训练 + 微调」的全面应用。

---

## 一、核心创新：{% include gloss.html term="MLM" %}

{% include gloss.html term="BERT" %} 的最核心创新是 **{% include gloss.html term="MLM" %}（掩码语言模型）**。

### 为什么需要 MLM？

在 {% include gloss.html term="BERT" %} 之前，{% include gloss.html term="GPT" %} 使用自回归（从左到右）方式预训练。这种方式有天然缺陷：**只能利用单向上下文**。

```
自回归（单向）的问题：
  
  句子: "我去___场看电影"
  
  GPT 处理到 ___ 时，只能看到 "我去"
  → 不知道后面是 "场看电影"
  → 预测难度大，信息利用不充分 ❌
  
  BERT 的 MLM 方式：
  同时看到 "我去" 和 "场看电影"
  → 完整上下文信息 → 预测更准确 ✅
```

### MLM 具体做法

随机选择 15% 的 token 进行掩码处理，三种替换策略：

```
80% → [MASK]:      "我去 [MASK] 场看电影"
10% → 随机词:       "我去 操 场看电影"  ← 迫使模型依赖上下文
10% → 保持不变:     "我去 电 场看电影"  ← 迫使模型输出原词
```

### 为什么三种策略同时用？

如果全部用 `[MASK]`，模型只在预训练时见到 `[MASK]`，微调时却从来没见过。混合策略强制模型：

1. 当看到 `[MASK]` 时 → 从上下文推理正确词
2. 当看到真实词时 → 仍要正确编码该词（不能偷懒）
3. 当看到错词时 → 知道这个词不合理，用上下文纠正

### 下一句预测（NSP）

除了 {% include gloss.html term="MLM" %}，{% include gloss.html term="BERT" %} 还有一个辅助任务：**预测两个句子是否连续**。

```
输入: [CLS] 我去看电影 [SEP] 电影很好看 [SEP]  → 连续 → 标签: IsNext
输入: [CLS] 我去看电影 [SEP] 苹果很好吃 [SEP]  → 不连续 → 标签: NotNext
```

这个任务让 BERT 理解句子间关系，对 QA、推理等任务有帮助。

---

## 二、BERT 架构

{% include gloss.html term="BERT" %} 使用 **Transformer 编码器**架构：

```
BERT Base（1.1 亿参数）:
  12 层 Transformer 编码器
  768 隐藏维度
  12 注意力头
  训练数据: BookCorpus + Wikipedia (3.3B 词)

BERT Large（3.4 亿参数）:
  24 层 Transformer 编码器
  1024 隐藏维度
  16 注意力头
  训练数据: BookCorpus + Wikipedia (3.3B 词)
```

### 与 GPT 的架构对比

```
GPT（解码器-only）:
  输入 → [掩码自注意力] → [FFN] → ... → 输出
         ↑ 因果掩码，只能从左到右

BERT（编码器-only）:
  输入 → [双向自注意力] → [FFN] → ... → 输出
         ↑ 无掩码，所有位置互相可见
```

### 输入表示

```
输入:  [CLS] 我 爱 [MASK] 学 习 [SEP]  它 很 有 趣 [SEP]

Token Embeddings:    每个词映射为向量
Segment Embeddings:  区分 A 句(0) 和 B 句(1)
Position Embeddings: 位置编码

三者相加 → BERT 的输入
```

`[CLS]` 位置的输出被用作整个句子的表示，用于分类任务。

---

## 三、预训练 + 微调范式

{% include gloss.html term="BERT" %} 的开创性不仅在于架构，还在于它推广了「预训练 + 微调」范式：

```
预训练阶段（一大步）:
  互联网文本 → BERT（MLM + NSP）→ 通用语言表示
  ↑ 一次训练，通用

微调阶段（一小步）:
  通用 BERT → 添加任务头 → 在特定任务上微调
  ↑ 少量标注数据，快速适配
```

### 微调示例

```
情感分类:
  BERT → [CLS]输出 → Linear(768, 2) → 正面/负面
  标注 1000 条 → 微调 1 小时 → 高精度

命名实体识别:
  BERT → 每个位置输出 → Linear(768, N) → 实体标签
  标注 2000 条 → 微调 2 小时 → 高精度

问答系统:
  BERT → 输出 → 预测答案起始/结束位置
  标注 5000 条 → 微调 3 小时 → 高精度
```

---

## 四、BERT 的影响

### BERT 发布时的 11 项 SOTA

| 任务 | 之前最佳 | BERT | 提升 |
|------|---------|------|------|
| GLUE 综合 | 80.2 | **86.5** | +6.3 |
| SQuAD 1.1 (QA) | 87.4 | **93.2** | +5.8 |
| SQuAD 2.0 (QA) | 80.2 | **86.8** | +6.6 |
| SWAG (推理) | 80.3 | **86.4** | +6.1 |

### 为什么 BERT 如此重要

1. **双向预训练被验证有效**——MLM 比自回归更适合理解任务
2. **「预训练+微调」成为标准范式**——BERT 之后的新模型几乎都采用此范式
3. **BERT 的变体层出不穷**——RoBERTa、ALBERT、DistilBERT、SpanBERT
4. **BERT 启发了检索模型**——Sentence-BERT、DPR 等

虽然 {% include gloss.html term="GPT" %} 系列后来在生成任务上胜出，但 BERT 的双向理解思想依然是 NLP 的重要遗产。

---

## 五、{% include gloss.html term="GPT" %} vs {% include gloss.html term="BERT" %}：最终对比

| 维度 | GPT | BERT |
|------|-----|------|
| 架构 | 解码器-only | 编码器-only |
| 注意力 | 单向（因果掩码）| **双向** |
| 预训练任务 | 自回归 | **MLM + NSP** |
| 适合任务 | 文本生成 | **自然语言理解** |
| 推理 | 逐个生成（慢）| 一次编码（快）|
| 开源 | 不完全 | **完全开源** |
| 发展 | GPT-3/4 统治生成 | BERT 变体统治理解 |

> 两者的共同奠基人：{% include gloss.html term="Transformer" %} 架构。没有 2017 年的 Transformer，就没有 GPT 和 BERT 的辉煌。

---

## 参考文献

- [BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding](https://arxiv.org/abs/1810.04805)
- [RoBERTa: A Robustly Optimized BERT Pretraining Approach](https://arxiv.org/abs/1907.11692)
- [ALBERT: A Lite BERT for Self-supervised Learning of Language Representations](https://arxiv.org/abs/1909.11942)
