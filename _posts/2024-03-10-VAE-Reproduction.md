---
layout: post
title: VAE 变分自编码器详解：从零推导到代码实现
categories: paper-reproduction
tags: [VAE, PyTorch, 生成模型, 入门]
---

> VAE（Variational Autoencoder）是生成模型的基石之一。这篇文章从"为什么要生成"开始，一步步推导出 VAE 的完整数学形式和代码实现。

---

## 一、生成模型要解决什么问题？

### 核心问题：学习数据的分布

假设你有 10000 张猫的图片。**生成模型**的目标是：学完这些图片后，能**生成全新的、看起来像猫的图片**。

```
训练集: [🐱₁, 🐱₂, 🐱₃, ..., 🐱₁₀₀₀₀]  →  学分布 p(x)

生成: 采样自 p(x) → 得到全新的 🐱 (训练集中没有的)
```

### 核心难点：高维空间的稀疏性

图片是**高维数据**。一张 28×28 的灰度图（MNIST）是 784 维空间中的一个点。要直接学习 $p(x)$ 在 784 维空间的分布，需要**指数级**的数据量——这被称为**维度灾难**。

### VAE 的思路：先降维再生成

VAE 的核心洞察：虽然 $x$ 在超高维空间，但它可以被一个**低维的潜在变量 $z$** 来解释。

```
  z（低维潜在空间）             x（高维数据空间）
  ┌─────────┐               ┌──────────────┐
  │  "猫性"  │               │              │
  │   • 耳朵形状  │──解码──→ │  🐱₁  🐱₂  🐱₃ │
  │   • 毛色     │  p(x|z)   │  🐱₄  🐱₅  🐱₆ │
  │   • 眼睛大小  │           │              │
  └─────────┘               └──────────────┘
  维度 ≈ 20                 维度 = 784（28×28）
```

---

## 二、{% include gloss.html term="VAE" %} 架构

### 与普通自编码器的关键区别

```
普通自编码器（Autoencoder）:
  输入 x → [编码器] → z（一个固定向量）→ [解码器] → 重建 x̂

  问题：z 是"一个点"，无法从中采样生成新数据
        → 只能重建见过的数据

VAE:
  输入 x → [编码器] → μ, σ²（一个分布的参数）→ 采样 z → [解码器] → 重建 x̂
                      ↓
                  潜在分布 q(z|x)
  
  优势：z 从分布中采样 → 可以生成无穷多新数据！
```

### 架构图

```
         编码器（推断网络）          潜在空间             解码器（生成网络）

  x ──→ [NN] ──→ μ ─────┐
         │               ├──→ z = μ + σ·ε ──→ [NN] ──→ x̂
         └──→ σ² ──────┘          ↑
                                  ε ∼ N(0, I)  ← 随机噪声
```

### 三个核心组件

**1. 编码器（Encoder / 推断网络）**

输入 $x$，输出潜在分布 $q(z|x)$ 的参数 $(\mu, \sigma^2)$。

```python
def encode(self, x):
    h = self.encoder(x)     # 共享特征提取
    mu = self.mu_layer(h)   # 均值
    logvar = self.logvar_layer(h)  # 对数方差（数值稳定）
    return mu, logvar
```

**2. 潜在空间（Latent Space）**

从 $N(\mu, \sigma^2)$ 中采样一个 $z$。这个 $z$ 代表了输入 $x$ 的"压缩表示"。

**3. 解码器（Decoder / 生成网络）**

从 $z$ 重建出 $x$。训练好后，可以直接从 $z \sim N(0, I)$ 采样来生成新数据。

---

## 三、损失函数：两个目标的平衡

VAE 的损失由两部分组成：**重构损失 + KL 散度**。

### 1. 重构损失（Reconstruction Loss）

让生成的图片和原图尽可能相似：

$$ \mathcal{L}_{\text{recon}} = -\mathbb{E}_{z \sim q(z|x)}[\log p(x|z)] $$

**直观理解**：给定潜在变量 $z$，生成原始数据 $x$ 的概率有多大？概率越大，损失越小。

对于图像，常用**二值交叉熵（BCE）**：

```python
recon_loss = F.binary_cross_entropy(recon_x, x, reduction='sum')
# 每个像素的交叉熵之和
```

### 2. {% include gloss.html term="KL Divergence" %} 正则化

强制 $q(z|x)$ 接近标准正态分布 $N(0, I)$：

$$ D_{\text{KL}}(q(z|x) \| p(z)) = \int q(z|x) \log \frac{q(z|x)}{p(z)} dz $$

对于两个高斯分布，有**闭合解**：

$$ D_{\text{KL}} = -\frac{1}{2} \sum_{j=1}^{J} \left(1 + \log \sigma_j^2 - \mu_j^2 - \sigma_j^2\right) $$

```python
kl_loss = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
```

**KL 散度的作用**：

```
KL 散度大 → q(z|x) 偏离 N(0, I) 太远
           → 潜在空间结构不好 → 采样出来的 z 没意义 ❌

KL 散度小 → q(z|x) 接近 N(0, I)  
           → 潜在空间规整 → 可以从 N(0, I) 采样生成 ✅

但 KL 散度太小也不行 → 会忽略 x 的信息 → 重建效果差
```

### 3. 总损失：两个目标的平衡

$$ \mathcal{L} = \underbrace{-\mathbb{E}_{z \sim q(z|x)}[\log p(x|z)]}_{\text{重构损失（逼真性）}} + \underbrace{D_{\text{KL}}(q(z|x) \| p(z))}_{\text{KL 散度（可生成性）}} $$

```
平衡关系：
  重构损失小 → 重建质量高 → 但可能过拟合（只会复制）
  KL 散度小 → 潜在空间规整 → 但可能忽略 x 的信息
  
  好的 VAE → 两者平衡 → 既能重建又能生成
```

---

## 四、{% include gloss.html term="Reparameterization Trick" %}：为什么需要它

### 问题：采样不可微

VAE 训练中最棘手的障碍：

```
编码器 → μ ──┐
              ├──→ z ∼ N(μ, σ²) → 解码器
编码器 → σ² ─┘
              ↑
        采样操作！梯度无法反向传播 ❌
```

### 解决方法

{% include gloss.html term="Reparameterization Trick" %} 把随机采样拆成两步：**确定性变换 + 独立噪声**。

$$
z = \mu + \sigma \cdot \varepsilon, \quad \varepsilon \sim \mathcal{N}(0, I)
$$

```
原来（不可微）:
  z = sample_from(N(μ, σ²))   ← 采样操作无梯度 ❌

现在（可微）:
  z = μ + σ × ε              ← 纯代数运算 ✅
       ↑     ↑
     可微    ε ∼ N(0,I) 独立于模型参数
```

**为什么可行？**

因为 $\varepsilon$ 是与模型参数无关的随机噪声，$\mu$ 和 $\sigma$ 是由编码器输出的确定性函数。梯度可以沿 $\mu$ 和 $\sigma$ 回传到编码器。

```python
def reparameterize(self, mu, logvar):
    """重参数化技巧"""
    std = torch.exp(0.5 * logvar)       # 标准差 σ
    eps = torch.randn_like(std)          # 独立噪声 ε ∼ N(0, I)
    z = mu + eps * std                   # z = μ + σ × ε
    return z
```

---

## 五、完整代码实现

```python
import torch
import torch.nn as nn
import torch.nn.functional as F

class VAE(nn.Module):
    def __init__(self, input_dim=784, latent_dim=20):
        super().__init__()
        # === 编码器 ===
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 400),
            nn.ReLU()
        )
        self.mu_layer = nn.Linear(400, latent_dim)       # 均值
        self.logvar_layer = nn.Linear(400, latent_dim)    # 对数方差

        # === 解码器 ===
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, 400),
            nn.ReLU(),
            nn.Linear(400, input_dim),
            nn.Sigmoid()  # 确保输出在 [0, 1] 范围
        )

    def encode(self, x):
        h = self.encoder(x)
        return self.mu_layer(h), self.logvar_layer(h)

    def reparameterize(self, mu, logvar):
        # z = μ + σ × ε, ε ∼ N(0,I)
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def decode(self, z):
        return self.decoder(z)

    def forward(self, x):
        mu, logvar = self.encode(x)          # 编码 → 分布参数
        z = self.reparameterize(mu, logvar)  # 重参数化采样
        recon = self.decode(z)               # 解码重建
        return recon, mu, logvar


def vae_loss(recon_x, x, mu, logvar):
    """VAE 损失函数"""
    # 重构损失：二值交叉熵
    recon_loss = F.binary_cross_entropy(
        recon_x, x, reduction='sum'
    )

    # KL 散度：闭合解
    # D_KL = -0.5 * Σ(1 + log(σ²) - μ² - σ²)
    kl_loss = -0.5 * torch.sum(
        1 + logvar - mu.pow(2) - logvar.exp()
    )

    return recon_loss + kl_loss
```

---

## 六、训练循环

```python
# 超参数
input_dim = 784      # 28×28 图片展平
latent_dim = 20      # 潜在空间维度
batch_size = 128
lr = 1e-3
num_epochs = 50

# 初始化
model = VAE(input_dim, latent_dim)
optimizer = torch.optim.Adam(model.parameters(), lr=lr)

# 训练
for epoch in range(num_epochs):
    total_loss = 0
    for batch_idx, (data, _) in enumerate(train_loader):
        data = data.view(-1, input_dim)  # 展平图片
        optimizer.zero_grad()

        # 前向传播
        recon_batch, mu, logvar = model(data)

        # 计算损失
        loss = vae_loss(recon_batch, data, mu, logvar)

        # 反向传播
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    avg_loss = total_loss / len(train_loader.dataset)
    print(f'Epoch {epoch:3d} | 平均损失: {avg_loss:.2f}')
```

---

## 七、生成新数据

训练完成后，生成新数据只需解码器部分：

```python
def generate(model, num_samples=16, latent_dim=20):
    """从标准正态分布采样，生成新数据"""
    model.eval()
    with torch.no_grad():
        # 从 N(0, I) 采样
        z = torch.randn(num_samples, latent_dim)
        # 解码生成
        samples = model.decode(z)
    return samples
```

### 潜在空间的插值

VAE 最强大的特性之一：**平滑插值**。

```
z₁ = encode("猫 A")    →  猫 A 的潜在表示
z₂ = encode("猫 B")    →  猫 B 的潜在表示

z_α = (1-α) × z₁ + α × z₂  →  中间的潜在表示

α = 0:    猫 A
α = 0.25: 偏 A 的中间猫
α = 0.5:  猫 A 和猫 B 的平均
α = 0.75: 偏 B 的中间猫
α = 1:    猫 B
```

这是因为 KL 散度强制潜在空间连续且规整——邻近的点对应语义相似的内容。

---

## 八、VAE vs 其他生成模型

| 特性 | VAE | GAN | Diffusion |
|------|-----|-----|-----------|
| 训练稳定性 | ✅ 稳定 | ❌ 易崩溃 | ✅ 稳定 |
| 生成质量 | ⭐⭐⭐ | ⭐⭐⭐⭐⭐ | ⭐⭐⭐⭐⭐ |
| 多样性 | ⭐⭐⭐⭐⭐ | ⭐⭐⭐ | ⭐⭐⭐⭐ |
| 潜在空间可解释性 | ✅ ✅ 好 | ❌ 差 | ⭐⭐ |
| 推理速度 | 快 | 快 | **慢** |
| 数学推导 | 优美 | 博弈论 | 复杂 |

---

## 九、总结

| 概念 | 一句话理解 |
|------|-----------|
| {% include gloss.html term="VAE" %} | 学习数据的潜在分布，而不是直接学数据本身 |
| 编码器 | 把数据压缩成分布参数 (μ, σ²) |
| 解码器 | 从潜在变量 z 重建/生成数据 |
| {% include gloss.html term="KL Divergence" %} | 正则化项，让潜在空间规整有序 |
| {% include gloss.html term="Reparameterization Trick" %} | 让采样可微，使端到端训练成为可能 |

### 扩展阅读

- **β-VAE**：加重 KL 权重，让潜在表示更解耦
- **VQ-VAE**：用离散潜在空间替代连续空间，生成质量更好
- **Conditional VAE**：加入条件输入（如类别标签），可控生成

---

## 参考文献

- [Auto-Encoding Variational Bayes, ICLR 2014](https://arxiv.org/abs/1312.6114)
- [Tutorial on Variational Autoencoders](https://arxiv.org/abs/1606.05908)
- [β-VAE: Learning Basic Visual Concepts with a Constrained Variational Framework](https://openreview.net/forum?id=Sy2fzU9gl)
