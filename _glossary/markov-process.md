---
layout: glossary
title: Markov Process
categories: [数学基础]
---

**Markov Process**（马尔可夫过程）是一种随机过程，其核心特性是**无后效性**（马尔可夫性）：**未来状态仅依赖于当前状态，与过去状态无关**。

## 数学定义

$$
P(X_{t+1} \mid X_t, X_{t-1}, \dots, X_0) = P(X_{t+1} \mid X_t)
$$

## 在扩散模型中的应用

扩散模型的前向加噪过程是一个典型的马尔可夫过程：

$$
q(x_t \mid x_{t-1}, x_{t-2}, \dots, x_0) = q(x_t \mid x_{t-1})
$$

每一步加噪只依赖于上一步的结果，与更早的状态无关。这一性质带来了两个关键好处：

1. **计算高效**：可以"跳跃"计算任意时间步的状态（无需逐 step 迭代）
2. **推导简洁**：联合分布可以分解为条件分布的连乘

## 跳跃式加噪的数学原理

利用马尔可夫性质，我们可以直接从 $x_0$ 计算 $x_t$：

$$
x_t = \sqrt{\bar{\alpha}_t} x_0 + \sqrt{1-\bar{\alpha}_t} \epsilon
$$

其中 $\bar{\alpha}_t = \prod_{i=1}^t \alpha_i$。

## 相关概念

- [扩散模型](/glossary/diffusion-model/)
- [DDPM](/glossary/ddpm/)
