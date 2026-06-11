---
layout: glossary
title: DDPM
categories: [生成模型]
---

**DDPM**（Denoising Diffusion Probabilistic Models，去噪扩散概率模型）是扩散模型的奠基性工作，由 Ho et al. 于 2020 年提出。

## 核心贡献

1. 将扩散模型的前向加噪和逆向去噪过程系统化、公式化
2. 提出简化的训练目标：直接预测噪声 $\epsilon$
3. 证明了简化损失函数等价于优化变分下界

## 训练过程

```
1. 从数据中采样 x₀
2. 随机选择时间步 t
3. 采样噪声 ε ∼ N(0, I)，生成 x_t
4. 训练模型 ε_θ 预测噪声 ε
5. 损失函数: ||ε - ε_θ(x_t, t)||²
```

## 采样过程

从纯噪声 $x_T \sim \mathcal{N}(0, I)$ 开始，逐步去噪：

$$
x_{t-1} = \frac{1}{\sqrt{\alpha_t}} \left(x_t - \frac{1-\alpha_t}{\sqrt{1-\bar{\alpha}_t}} \epsilon_\theta(x_t, t)\right) + \sigma_t z
$$

## 相关概念

- [扩散模型](/glossary/diffusion-model/)
- [马尔可夫过程](/glossary/markov-process/)
- [高斯噪声](/glossary/gaussian-noise/)
