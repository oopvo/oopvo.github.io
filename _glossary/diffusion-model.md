---
layout: glossary
title: Diffusion Model
categories: [生成模型]
---

**Diffusion Model**（扩散模型）是一类受热力学启发的生成模型，通过逐步加噪破坏数据分布，再学习逆向去噪过程来生成数据。

## 核心思想

扩散模型定义了两个过程：

1. **前向扩散过程**（加噪）：逐步向数据添加高斯噪声，直到数据变为纯噪声
2. **逆向去噪过程**（去噪）：训练神经网络从纯噪声中逐步恢复出原始数据

## 关键特性

- 生成质量高，在图像生成领域达到甚至超越 GAN
- 训练稳定，不存在 GAN 的对抗训练不收敛问题
- 理论基础扎实，有清晰的数学推导
- 缺点是采样速度较慢（需要多步迭代）

## 代表模型

- **DDPM**（Denoising Diffusion Probabilistic Models）— 奠基之作
- **DDIM**（Denoising Diffusion Implicit Models）— 加速采样
- **Stable Diffusion** — 在潜在空间做扩散，大幅降低计算量

## 相关概念

- [DDPM](/glossary/ddpm/)
- [马尔可夫过程](/glossary/markov-process/)
- [高斯噪声](/glossary/gaussian-noise/)
- [VAE](/glossary/vae/)
