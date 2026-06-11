---
layout: glossary
title: Gaussian Noise
categories: [数学基础]
---

**Gaussian Noise**（高斯噪声）指服从正态分布（高斯分布）的随机噪声，是扩散模型中使用的核心噪声类型。

## 定义

若随机变量 $\epsilon$ 服从均值为 $\mu$、方差为 $\sigma^2$ 的高斯分布，记作：

$$
\epsilon \sim \mathcal{N}(\mu, \sigma^2 I)
$$

标准高斯噪声取 $\mu = 0,\ \sigma^2 = 1$：

$$
\epsilon \sim \mathcal{N}(0, I)
$$

## 在扩散模型中的作用

- 前向过程使用高斯噪声逐步破坏图像
- 模型训练的目标是预测所加的高斯噪声
- 逆向过程的每一步也是高斯分布（高斯分布经线性变换后仍是高斯分布）

## 重参数化技巧

高斯噪声的一个优良性质：任意的 $\mathcal{N}(\mu, \sigma^2)$ 均可通过标准高斯噪声线性变换得到：

$$
z = \mu + \sigma \cdot \epsilon, \quad \epsilon \sim \mathcal{N}(0, I)
$$

这使得梯度可以反向传播到 $\mu$ 和 $\sigma$。

## 相关概念

- [扩散模型](/glossary/diffusion-model/)
- [DDPM](/glossary/ddpm/)
- [重参数化技巧](/glossary/reparameterization-trick/)
