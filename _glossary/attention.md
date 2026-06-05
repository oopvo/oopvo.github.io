---
layout: glossary
title: Attention
categories: [模型架构]
---

**Attention**（注意力机制）是一种让模型在处理每个位置时动态关注输入序列中其他位置的机制。

## 核心计算

```
Attention(Q, K, V) = softmax(QK^T / √d_k) × V

Q (Query):    当前位置的查询向量
K (Key):      所有位置的键向量  
V (Value):    所有位置的值向量
d_k:          缩放因子，防止内积过大
```

## 为什么需要 Attention

传统序列模型（RNN）只能顺序处理，长距离信息需要通过多个时间步传递，容易丢失。Attention 让模型**一步到位**建立任意两个位置之间的直接连接。

## Scaled Dot-Product Attention

这是最常用的注意力实现方式：
1. 计算 Q 和 K 的点积 → 相似度分数
2. 除以 √d_k 缩放 → 防止 softmax 梯度过小
3. softmax 归一化 → 注意力权重
4. 加权求和 V → 输出
