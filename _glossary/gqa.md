---
layout: glossary
title: GQA
categories: [模型架构]
related_posts:
  - Llama 系列技术报告精读：Meta 的开源大模型之路
---

**Grouped-Query Attention**（分组查询注意力）是 MHA（Multi-Head Attention）和 MQA（Multi-Query Attention）的折中方案。

## 演进对比

```
MHA（标准多头注意力）:
  Q: 32 头    K: 32 头    V: 32 头
  KV 缓存 = 32 × dim
  质量最高，缓存最大

GQA（分组查询注意力）:
  Q: 32 头    K: 8 组     V: 8 组
  KV 缓存 = 8 × dim
  质量与 MHA 接近，缓存减少 75%

MQA（单查询注意力）:
  Q: 32 头    K: 1 头     V: 1 头
  KV 缓存 = 1 × dim
  缓存最小，质量略降
```

## 优势

GQA 在保持接近 MHA 质量的同时，大幅减少 KV 缓存和访存开销，是现代大模型（Llama 2/3/4、Gemma）的标配注意力机制。
