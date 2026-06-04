---
layout: post
title: 论文复刻 — VAE 变分自编码器 PyTorch 实现
categories: paper-reproduction
tags: [VAE, PyTorch, 生成模型]
---

## 原论文

- **Auto-Encoding Variational Bayes** (Kingma & Welling, ICLR 2014)

## 核心公式

VAE 的损失函数由两部分组成：

1. **重构损失**（Reconstruction Loss）：衡量生成数据与原始数据的差异
2. **KL 散度**（KL Divergence）：衡量近似后验与先验分布的差异

$$
\mathcal{L}(\theta, \phi; x) = -D_{KL}(q_\phi(z|x) \| p_\theta(z)) + \mathbb{E}_{q_\phi(z|x)}[\log p_\theta(x|z)]
$$

## 代码实现

```python
import torch
import torch.nn as nn

class VAE(nn.Module):
    def __init__(self, input_dim=784, latent_dim=20):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 400),
            nn.ReLU()
        )
        self.mu = nn.Linear(400, latent_dim)
        self.logvar = nn.Linear(400, latent_dim)
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, 400),
            nn.ReLU(),
            nn.Linear(400, input_dim),
            nn.Sigmoid()
        )
    
    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std
    
    def forward(self, x):
        h = self.encoder(x)
        mu, logvar = self.mu(h), self.logvar(h)
        z = self.reparameterize(mu, logvar)
        return self.decoder(z), mu, logvar
```
