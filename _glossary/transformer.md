---
layout: glossary
title: Transformer
categories: [模型架构]
---

**Transformer** 是一种基于自注意力机制的深度神经网络架构，2017 年由 Google 在论文 "Attention Is All You Need" 中提出。它摒弃了传统的循环/卷积结构，完全依赖注意力机制建模序列数据中的长距离依赖关系，奠定了现代大语言模型的基础。

## 完整架构图

![Transformer 完整架构（Harvard NLP - 原论文图）](https://github.com/harvardnlp/annotated-transformer/raw/master/images/ModalNet-21.png)

```
输出序列
    ▲
    │
┌───┴──────────────────┐
│    Linear + Softmax   │  ← 映射到词表概率
└───┬──────────────────┘
    │
┌───┴──────────────────┐
│    Add & Norm         │
│    Feed Forward       │  ← 两层线性变换 + ReLU
│    (position-wise)    │
└───┬──────────────────┘
    │
┌───┴──────────────────┐
│    Add & Norm         │
│    Multi-Head         │  ← 多头自注意力（8 个头）
│    Attention          │
└───┬──────────────────┘      ↑ 如果是 Decoder，这里还有 Cross-Attention
    │                           │
    │  + Positional Encoding    │
    │                           │
输入序列 ───────────────────────┘
```

## 核心组件

- **{% include gloss.html term="Attention" %}**：每个位置关注所有位置，一步建立长距离连接
- **{% include gloss.html term="Multi-Head Attention" %}**：8 个头并行，从不同子空间关注信息
- **{% include gloss.html term="Positional Encoding" %}**：注入位置信息，解决置换不变性
- **前馈网络（FFN）**：每个位置独立的非线性变换
- **残差连接 + 层归一化**：稳定深层网络训练

## 输入输出流程

```
输入: "我 爱 学习"
  1. Token Embedding: 每个词映射为 512 维向量
  2. + Positional Encoding: 加入位置信号
  3. Multi-Head Self-Attention: 每个词关注其他词
  4. Feed Forward: 逐位置非线性变换
  5. × 6 层: 重复上述过程
  6. Output Projection: 映射到词表
输出: "I love learning"
```
