---
layout: post
title: Transformer 架构详解：从 Attention 到 Multi-Head
categories: deep-learning
tags: [Transformer, Attention, NLP]
---

Transformer 架构自 2017 年提出以来，已经成为自然语言处理乃至整个深度学习领域最重要的基础架构之一。

本文将从最基础的 Attention 机制开始，逐步深入到 Multi-Head Attention、Positional Encoding 等核心组件，帮助读者全面理解 Transformer 的设计思想与技术细节。

## Attention 机制

Attention 的核心思想是让模型在处理每个位置时，能够关注到输入序列中的所有其他位置，从而捕捉长距离依赖关系。

### Scaled Dot-Product Attention

计算公式如下：

$$
\text{Attention}(Q, K, V) = \text{softmax}\left(\frac{QK^T}{\sqrt{d_k}}\right) V
$$

## Multi-Head Attention

Multi-Head Attention 将查询、键、值分别投影到多个子空间，在不同的子空间中并行计算 Attention，最后将结果拼接起来。
