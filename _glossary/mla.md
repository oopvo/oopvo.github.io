---
layout: glossary
title: MLA
categories: [模型架构]
related_posts:
  - DeepSeek-V3 技术报告精读：671B 参数的 MoE 奇迹
---

**Multi-head Latent Attention**（多头潜在注意力）是 DeepSeek-V3 的核心创新之一，主要解决大模型推理时 KV 缓存占用过大的问题。

## 解决的问题

标准 Transformer 在自回归生成时，需要缓存所有历史 token 的 Key 和 Value。对于 671B 模型，每 token 缓存 49,152 个元素，长上下文时显存爆炸。

## 核心思路

将 Key 和 Value 通过低秩投影压缩到一个 **512 维的潜在空间** 后再缓存，需要使用时再解压缩：

```
输入 hidden (7168-dim)
      │
      ▼
  W_dkv (Down-project)
      │
      ▼
  c_KV (512-dim)  ◀── 只缓存这个！
      │
  ┌───┴───┐
  ▼       ▼
W_uk    W_uv
  │       │
  ▼       ▼
K       V
  │       │
  └───┬───┘
      ▼
  Attention
```

## 效果

- KV 缓存减少 **98.6%**
- 每 token 从 49,152 元素降至 576 元素
- 单 GPU 可处理 **1,149,440 tokens** KV 缓存
- 支持 512+ 并发请求

## 吸收模式

在推理时，解压缩矩阵 `W_uk`、`W_uv` 被吸收进 Q 投影矩阵，注意力直接在潜在空间计算，实现 **71× 内存效率提升**。
