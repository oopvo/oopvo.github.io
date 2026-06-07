---
layout: glossary
title: BERT
categories: [模型架构]
---

**Bidirectional Encoder Representations from Transformers**（双向编码器表示）是 Google 于 2018 年提出的预训练语言模型。

## 核心创新：双向预训练

与 {% include gloss.html term="GPT" %} 自回归（从左到右）不同，BERT 使用 **{% include gloss.html term="MLM" %}（掩码语言模型）** 实现双向理解：

```
传统自回归（GPT）:
  我 → 爱 → [？] → 学 → 习
  → 只能看左边 → 单向 ❌

BERT 双向 MLM:
  我 → 爱 → [MASK] → 学 → 习
  → 同时看左右 → 双向 ✅
  
  模型需要预测 [MASK] = "深度"
```

## 架构

BERT 使用 **Transformer 编码器**（双向自注意力 + 无因果掩码）：

```
输入: [CLS] I love [MASK] learning [SEP]
    │
    ▼
┌──────────────────┐
│ Token + Segment  │
│ + Position Embed │
└──────┬───────────┘
       │
┌──────┴───────────┐
│ Bidirectional    │  ← 无掩码！所有位置互相可见
│ Multi-Head Attn  │
└──────┬───────────┘
       │  × N 层
       ▼
输出: 每个位置的 contextual embedding
```

## 影响

BERT 在发布时在 **11 项 NLP 基准上取得 SOTA**，开启了「预训练 + 微调」范式的全面应用。其编码器架构虽然被后来的生成模型（GPT）取代，但双向理解的思想深刻影响了后续模型设计。
