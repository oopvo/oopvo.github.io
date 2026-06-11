---
layout: glossary
title: 信噪比
categories: [数学基础]
---

**信噪比**（Signal-to-Noise Ratio, SNR）衡量信号与噪声的强度之比，在扩散模型中用于描述每一步加噪后图像保留的信息量。

## 在扩散模型中的定义

在前向扩散过程中：

$$
x_t = \sqrt{\alpha_t} x_{t-1} + \sqrt{1-\alpha_t} \epsilon_{t-1}
$$

- **Signal**：$\sqrt{\alpha_t}$，控制上一时刻信息的保留比例
- **Noise**：$\sqrt{1-\alpha_t}$，控制新加入噪声的强度

信噪比定义为：

$$
\text{SNR}(t) = \frac{\alpha_t}{1-\alpha_t}
$$

## 调度策略

随着 $t$ 增大，$\alpha_t$ 递减，信噪比递减，表示图像信息逐渐被噪声淹没，最终 $t=T$ 时近似为纯噪声。

## 相关概念

- [扩散模型](/glossary/diffusion-model/)
- [高斯噪声](/glossary/gaussian-noise/)
