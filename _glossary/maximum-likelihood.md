---
layout: glossary
title: 最大似然估计
categories: [训练方法]
---

**最大似然估计**（Maximum Likelihood Estimation, MLE）是一种参数估计方法，核心思想是：**找到一组参数 $\theta$，使观测数据出现的概率最大**。

## 数学形式

给定观测数据 $X = \{x_1, x_2, \dots, x_n\}$，似然函数为：

$$
\mathcal{L}(\theta \mid X) = \prod_{i=1}^n p_\theta(x_i)
$$

最大似然估计即最大化该函数（通常取对数方便计算）：

$$
\theta_{\text{MLE}} = \arg\max_\theta \log \mathcal{L}(\theta \mid X)
$$

## 在扩散模型中的应用

扩散模型的训练目标可以看作最大化数据的对数似然（或其变分下界 ELBO），等价于**最小化预测噪声与真实噪声之间的 MSE**。

## 与 KL 散度的关系

最大化似然等价于最小化数据分布 $p_{\text{data}}$ 与模型分布 $p_\theta$ 之间的 KL 散度。

## 相关概念

- [扩散模型](/glossary/diffusion-model/)
- [DDPM](/glossary/ddpm/)
- [KL 散度](/glossary/kl-divergence/)
