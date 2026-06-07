---
layout: glossary
title: GPT
categories: [模型架构]
---

**Generative Pre-trained Transformer**（生成式预训练 Transformer）是 OpenAI 开发的大语言模型系列。

## 发展历程

```
GPT-1 (2018.06)  — 117M 参数，首次证明 Transformer 解码器预训练有效
    │
GPT-2 (2019.02)  — 1.5B 参数，展示零样本迁移能力
    │
GPT-3 (2020.05)  — 175B 参数，开创大模型时代
    │
GPT-4 (2023.03)  — 多模态，推理能力飞跃
```

## 核心架构：「解码器-only」

与 {% include gloss.html term="BERT" %} 的编码器架构不同，GPT 使用 **Transformer 解码器**（去掉交叉注意力）：

```
输入: "I love"
    │
    ▼
┌──────────────┐
│ Token Embed  │
│ + Positional │
└──────┬───────┘
       │
┌──────┴───────┐
│ Masked Multi-│  ← 只能看左侧（因果掩码）
│ Head Attn    │
└──────┬───────┘
       │
┌──────┴───────┐
│ Feed Forward │
└──────┬───────┘
       │  × N 层
       ▼
输出: "learning"（预测下一个词）
```

## 关键词

- **{% include gloss.html term="Autoregressive" %}**：逐个 token 自回归生成
- **大规模预训练**：互联网级数据训练通用表示
- **上下文学习（In-Context Learning）**：GPT-3 以来无需微调，通过 prompt 示例完成任务
