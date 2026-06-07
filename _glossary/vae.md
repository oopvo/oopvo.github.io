---
layout: glossary
title: VAE
categories: [模型架构]
---

**VAE**（Variational Autoencoder，变分自编码器）由 Kingma & Welling 于 2014 年提出，是一种生成模型。

![VAE 架构示意图](https://raw.githubusercontent.com/pclubiitk/model-zoo/master/generative_models/VAEGAN_PyTorch/assets/vae.png)

## 架构

```
                 编码器（推断）          潜在空间             解码器（生成）

输入 x ────→ [NN 编码器] ──→ μ ──────┐
  (784dim)               │          ├──→ z = μ + σ·ε ──→ [NN 解码器] ──→ x̂
                         └──→ σ² ────┘         ↑
                                               ε ∼ N(0, I)
                   
训练完成后，生成新数据只需解码器部分:
                  
  z ∼ N(0, I) ──→ [NN 解码器] ──→ 全新数据 🎨
  (从标准正态采样)      (训练好的)      (从未见过!)
```

## 损失函数

VAE 的损失由两部分组成：

1. **重构损失**：衡量生成数据与原始数据的差异（MSE 或交叉熵）
2. **KL 散度**：衡量近似后验 q(z|x) 与先验 p(z) 的差异

## 重参数化技巧

VAE 训练中的关键技巧：将「从分布采样」拆解为 **确定性变换 + 独立噪声**，使梯度可以反向传播。

```
不可微:          z = 从 N(μ, σ²) 采样
可微（重参数化）:  z = μ + σ × ε, ε ∼ N(0, I)
```
