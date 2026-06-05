---
layout: glossary
title: KL Divergence
categories: [训练方法]
---

**KL Divergence**（Kullback-Leibler 散度，KL 散度）衡量两个概率分布之间的差异。

## 在 VAE 中的作用

在 VAE 中，KL 散度作为**正则化项**，强制近似后验分布 q(z|x) 接近先验分布 p(z) = N(0, I)：

```
L = 重构损失 + KL(q(z|x) || p(z))

重构损失 → 让解码器精确重建输入
KL 散度  → 让潜在空间有良好的结构（连续、可插值）
```

## 直观理解

- KL 散度 = 0：q 和 p 完全相同
- KL 散度 越大：q 和 p 差异越大
- 在 VAE 中，KL 散度防止潜在空间过度发散

## 公式

```
KL(q||p) = ∫ q(z) log(q(z)/p(z)) dz
```

当 q 和 p 均为高斯分布时，有闭合解。
