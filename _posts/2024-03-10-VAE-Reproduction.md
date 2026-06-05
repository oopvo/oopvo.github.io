---
layout: post
title: 论文复刻 — VAE 变分自编码器 PyTorch 实现
categories: paper-reproduction
tags: [VAE, PyTorch, 生成模型]
---

## 原论文

- **Auto-Encoding Variational Bayes** (Kingma & Welling, ICLR 2014)

---

## VAE 的核心思想

**{% include gloss.html term="VAE" %}**（Variational Autoencoder，变分自编码器）是一种生成模型，它的目标不是简单地"压缩-重建"数据，而是学习数据的**潜在分布**，从而可以**生成新的数据**。

### 与普通自编码器的区别

```
普通自编码器:
  输入 → [编码器] → 固定向量 → [解码器] → 重建输入
                   （一个点）
  → 只能重建已有的数据

VAE:
  输入 → [编码器] → 均值 μ, 方差 σ² → 采样 z → [解码器] → 重建输入
                   （一个分布）
  → 可以从分布中采样，生成全新的数据
```

---

## 损失函数

{% include gloss.html term="VAE" %} 的损失函数由两部分组成：

### 1. 重构损失

衡量生成数据与原始数据的差异。对于图像，通常使用 **二值交叉熵**（BCE）或 **均方误差**（MSE）：

```
重构损失 = -E[log p(x|z)]
            ↑ 给定潜在变量 z，生成原始数据 x 的概率
```

### 2. {% include gloss.html term="KL Divergence" %} 正则化

{% include gloss.html term="KL Divergence" %} 强制近似后验分布 q(z|x) 接近先验分布 p(z) = N(0, I)：

```
KL 散度 = KL(q(z|x) || p(z))
         ↑ 衡量两个分布的差异

作用:
- 防止潜在空间过度发散
- 让潜在空间有良好的结构（连续、可插值）
- 使采样变得有意义
```

### 总损失

```
L = 重构损失 + KL(q(z|x) || p(z))
```

### 直观理解

```
重构损失 → 让解码器精确重建输入
          （逼真性）

KL 散度  → 让潜在空间保持规整
          （可生成性）

两者平衡：
  - 重构损失太小 → 过拟合，只能复制
  - KL 散度太小 → 潜在空间发散，采样没意义
  - 两者平衡 → 既能重建又能生成新数据
```

---

## {% include gloss.html term="Reparameterization Trick" %}

{% include gloss.html term="VAE" %} 训练中的关键技巧。问题是：**从分布采样是不可微的**，梯度无法通过采样节点回传到编码器。

### 问题示意

```
编码器 → μ ──┐
编码器 → σ² ──┤──→ 采样 z ∼ N(μ, σ²) → 解码器 → 输出
              │     ↑ 采样操作不可微！梯度堵在这里
```

### 解决方法

{% include gloss.html term="Reparameterization Trick" %} 将随机采样拆解为**确定性变换 + 独立随机噪声**：

```
编码器 → μ ──┐
             ├──→ z = μ + σ × ε → 解码器 → 输出
编码器 → σ ──┘          ↑
               ε ∼ N(0, I) 独立于模型参数
               ↑ 现在梯度可以沿 μ 和 σ 回传到编码器 ✅
```

```python
def reparameterize(mu, logvar):
    """重参数化技巧"""
    std = torch.exp(0.5 * logvar)          # 标准差
    eps = torch.randn_like(std)            # 独立噪声 ε ∼ N(0, I)
    z = mu + eps * std                     # z = μ + σ × ε
    return z
```

---

## 完整代码实现

```python
import torch
import torch.nn as nn
import torch.nn.functional as F

class VAE(nn.Module):
    def __init__(self, input_dim=784, latent_dim=20):
        super().__init__()
        # 编码器
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 400),
            nn.ReLU()
        )
        self.mu_layer = nn.Linear(400, latent_dim)
        self.logvar_layer = nn.Linear(400, latent_dim)

        # 解码器
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, 400),
            nn.ReLU(),
            nn.Linear(400, input_dim),
            nn.Sigmoid()
        )

    def encode(self, x):
        h = self.encoder(x)
        return self.mu_layer(h), self.logvar_layer(h)

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def decode(self, z):
        return self.decoder(z)

    def forward(self, x):
        mu, logvar = self.encode(x)         # 编码 → 分布参数
        z = self.reparameterize(mu, logvar) # 重参数化采样
        recon = self.decode(z)              # 解码重建
        return recon, mu, logvar


def vae_loss(recon, x, mu, logvar):
    """VAE 损失函数"""
    # 重构损失（二值交叉熵）
    recon_loss = F.binary_cross_entropy(
        recon, x, reduction='sum'
    )

    # KL 散度（闭合解，因为假设高斯分布）
    kl_loss = -0.5 * torch.sum(
        1 + logvar - mu.pow(2) - logvar.exp()
    )

    return recon_loss + kl_loss
```

---

## 训练循环

```python
model = VAE(input_dim=784, latent_dim=20)
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

for epoch in range(num_epochs):
    for batch_idx, (data, _) in enumerate(train_loader):
        data = data.view(-1, 784)
        optimizer.zero_grad()

        recon_batch, mu, logvar = model(data)
        loss = vae_loss(recon_batch, data, mu, logvar)

        loss.backward()
        optimizer.step()
```

---

## 对学习者的启示

1. **{% include gloss.html term="VAE" %}** 教会我们：生成的关键不是记忆数据，而是学习数据的**分布**
2. **{% include gloss.html term="KL Divergence" %}** 作为正则化项的思想在现代大模型训练中广泛使用
3. **{% include gloss.html term="Reparameterization Trick" %}** 是"让不可微变可微"的经典范式
4. {% include gloss.html term="VAE" %} 的变体（β-VAE、VQ-VAE）在图像生成、表示学习领域仍有广泛应用

---

## 参考文献

- [Auto-Encoding Variational Bayes, ICLR 2014](https://arxiv.org/abs/1312.6114)
- [Tutorial on Variational Autoencoders](https://arxiv.org/abs/1606.05908)
- [β-VAE: Learning Basic Visual Concepts with a Constrained Variational Framework](https://openreview.net/forum?id=Sy2fzU9gl)
