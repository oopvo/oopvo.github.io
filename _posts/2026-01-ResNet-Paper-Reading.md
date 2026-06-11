---
layout: post
title: 论文精读 — Deep Residual Learning for Image Recognition（未完善）
categories: paper-reading
tags: [ResNet, CV, 残差网络, 入门]
---

> 2015 年 **CVPR Best Paper**，被引 **200,000+ 次**。这篇论文用极其简洁的思想——"学习残差而不是学习目标"——解决了困扰深度学习多年的退化问题，让训练 1000+ 层网络成为可能。

---

## 一、核心问题：网络越深越差？

### 直觉：越深应该越强

直觉告诉我们：神经网络越深（层数越多），表达能力越强，效果应该越好。

```
理论上的期望：
  20 层网络 → 效果不错
  56 层网络 → 效果更好 ← 更多层 = 更强表达力
  100 层网络 → 效果最好
```

### 现实：{% include gloss.html term="Degradation Problem" %}

作者实验发现了一个反直觉的现象：

```
实际观察：
  20 层网络:   训练误差 = 0.10,  测试误差 = 0.15  ✅
  56 层网络:   训练误差 = 0.25,  测试误差 = 0.28  ❌ 更差！

关键：56 层的训练误差反而更高 → 这不是过拟合！
```

### 为什么层数多了反而更差？

这不是过拟合（overfitting）。过拟合时训练误差低、测试误差高。但这里**训练误差也高了**。

**根本原因：深层网络难以优化**

```
问题本质：
  给网络增加一层，这层最少应该做到"什么都不做"
  → 即输出等于输入（恒等映射 H(x) = x）
  
  但对 SGD（随机梯度下降）来说，学习"什么都不做"异常困难
  → 权重趋向于 0 比趋向于单位矩阵容易得多
  
  结果：额外层不仅没帮助，反而扰乱了已有表示
```

{% include gloss.html term="Degradation Problem" %} 的发现是这篇论文的起点：**不是网络越深越好，关键是怎么让深层网络能被有效优化**。

---

## 二、{% include gloss.html term="Residual Learning" %}：核心思想

### 换个思路：不学目标，学残差

作者的洞察极其简洁：**与其让网络学习完整的映射 H(x)，不如让它学习残差 F(x) = H(x) - x**。

### 数学变换

$$ H(x) = \text{目标映射} $$

**传统方式**：直接学习 $H(x)$

$$ \text{Loss} = \|H(x) - y\|^2 $$

当 $H(x)$ 应该是恒等映射时（即 $H(x) = x$），网络需要把所有权重调整为单位矩阵附近——这对于 SGD 来说很难。

**残差方式**：学习 $F(x) = H(x) - x$，然后 $H(x) = F(x) + x$

$$ F(x) = H(x) - x $$

当目标映射是 $H(x) = x$ 时，网络只需学习 $F(x) = 0$——**把所有权重趋向于 0 即可**。

### 为什么学习 F(x)=0 比 H(x)=x 容易？

```
传统学习 H(x) = x:
  权重初始化 ≈ N(0, 0.01)
  需要调整到单位矩阵附近
  → 权重需要大幅变化 → 困难 ❌

残差学习 F(x) = 0:
  权重初始化 ≈ N(0, 0.01) → 输出已经接近 0
  只需要把权重向 0 微调
  → 权重几乎不需要变化 → 容易 ✅
```

这就是残差学习的核心智慧：**调整学习目标，让优化变得简单**。

---

## 三、{% include gloss.html term="Shortcut Connections" %}：如何实现

### 残差块（Residual Block）

{% include gloss.html term="Shortcut Connections" %} 是实现 {% include gloss.html term="Residual Learning" %} 的具体方式：

```
输入 x
  │
  ├──────→ [Conv → BN → ReLU → Conv → BN] → F(x)
  │                                            │
  │         Shortcut Connection                │
  └────────────────────────────────────────────┘ ⊕
                                                    │
                                                 [ReLU]
                                                    │
                                                    输出
```

数学形式：

$$ y = \mathcal{F}(x, \{W_i\}) + x $$

其中 $\mathcal{F}(x, \{W_i\})$ 是残差映射（两个卷积层），$+ x$ 是快捷连接。

### 为什么叫"快捷"？

因为梯度可以**直接通过快捷连接回传**，不需要经过权重层：

```
反向传播时：
  损失 L
    │
  ∂L/∂y
    │
  ∂L/∂y = ∂L/∂x · (∂F/∂x + 1)  ← 关键：这里有一个 +1
                                  ← 梯度直接"抄近道"回来

  即使 ∂F/∂x 很小（权重层梯度消失），∂L/∂x 依然有 ∂L/∂y × 1 的量
  → 浅层的梯度不会消失！
```

### 代码实现

```python
import torch
import torch.nn as nn

class BasicBlock(nn.Module):
    """最基本的残差块：两个 3×3 卷积"""
    def __init__(self, in_channels, out_channels, stride=1):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, 3,
                               stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(out_channels, out_channels, 3,
                               padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)

        # 如果输入输出维度不同，需要 1×1 卷积调整维度
        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, 1,
                          stride=stride, bias=False),
                nn.BatchNorm2d(out_channels)
            )

    def forward(self, x):
        identity = self.shortcut(x)          # 保存输入（快捷连接）

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)

        out += identity                      # ✨ 残差连接！
        return self.relu(out)
```

---

## 四、网络架构

### ResNet 家族

| 层名 | 输出大小 | ResNet-18 | ResNet-34 | ResNet-50 | ResNet-101 | ResNet-152 |
|------|---------|-----------|-----------|-----------|------------|------------|
| conv1 | 112×112 | 7×7, 64, stride 2 | ← | ← | ← | ← |
| conv2_x | 56×56 | [3×3, 64] × 2 | [3×3, 64] × 3 | [1×1,64; 3×3,64; 1×1,256] × 3 | ×3 | ×3 |
| conv3_x | 28×28 | [3×3, 128] × 2 | [3×3, 128] × 4 | [1×1,128; 3×3,128; 1×1,512] × 4 | ×4 | ×8 |
| conv4_x | 14×14 | [3×3, 256] × 2 | [3×3, 256] × 6 | [1×1,256; 3×3,256; 1×1,1024] × 6 | ×23 | ×36 |
| conv5_x | 7×7 | [3×3, 512] × 2 | [3×3, 512] × 3 | [1×1,512; 3×3,512; 1×1,2048] × 3 | ×3 | ×3 |
| 总层数 | — | **18** | **34** | **50** | **101** | **152** |

### 两种残差块

**BasicBlock（18/34 层用）**：

```
         3×3, C        3×3, C
  x → [Conv-ReLU] → [Conv] → + → ReLU → 输出
  └──────────────────────────────────┘ (快捷连接)
```

**Bottleneck（50/101/152 层用）**：

```
        1×1, C/4      3×3, C/4      1×1, C
  x → [Conv-ReLU] → [Conv-ReLU] → [Conv] → + → ReLU → 输出
  └────────────────────────────────────────────────┘ (快捷连接)

  先降维再升维，减少计算量
```

---

## 五、实验结果

### ImageNet 分类

| 网络 | top-1 错误率 | top-5 错误率 | vs VGG |
|------|------------|------------|--------|
| VGG-16 | 28.5% | 9.3% | baseline |
| ResNet-34 | 26.7% | 8.5% | **↓ 1.8%** |
| ResNet-50 | 24.6% | 7.5% | **↓ 3.9%** |
| ResNet-101 | 23.6% | 7.1% | **↓ 4.9%** |
| **ResNet-152** | **23.0%** | **6.7%** | **↓ 5.5%** |

### 关键发现：更深没有退化

用 {% include gloss.html term="ResNet" %} 训练时，**更深网络的训练误差不再增加**：

```
普通网络（Plain-56）:  训练误差 0.25 ← 比 20 层还高（退化）
ResNet-56:             训练误差 0.08 ← 比 20 层更低 ✓
ResNet-110:            训练误差 0.06 ← 更深更好 ✓  （无退化！）
```

这验证了残差学习确实解决了 {% include gloss.html term="Degradation Problem" %}。

### ILSVRC 2015 夺冠

{% include gloss.html term="ResNet" %} 以 **3.57%** 的 top-5 错误率赢得 ILSVRC 2015 冠军（第二名 4.16%），这是**计算机视觉领域最重要的里程碑之一**。

---

## 六、{% include gloss.html term="ResNet" %} 的影响

### 为什么 ResNet 是里程碑？

1. **突破深度限制**：之前只能训练 10-20 层，ResNet 让 100+ 层网络变得可行
2. **思想简洁**：改动极其简单（加一条快捷连接），但效果极其显著
3. **即插即用**：残差连接不是 ResNet 的专利，几乎所有现代网络都在用

### 残差连接无处不在

```
Transformer:     output = LayerNorm(x + SelfAttention(x))  ← 残差连接！
CNN:             resblock 已成为标准组件
GAN:             残差连接提升训练稳定性
强化学习:         残差 Q 网络
```

{% include gloss.html term="Shortcut Connections" %} 从 {% include gloss.html term="ResNet" %} 开始，已成为深度学习的**基础设施**。

---

## 七、总结

| 概念 | 一句话 |
|------|--------|
| {% include gloss.html term="Degradation Problem" %} | 网络越深训练误差越高，不是过拟合，是优化困难 |
| {% include gloss.html term="Residual Learning" %} | 学残差 $F(x) = H(x) - x$ 比学目标 $H(x)$ 容易得多 |
| {% include gloss.html term="Shortcut Connections" %} | 一个加法操作，让梯度直接回传，深度不再是障碍 |
| {% include gloss.html term="ResNet" %} | 152 层，3.57% top-5 错误率，ILSVRC 2015 冠军 |

---

## 参考文献

- [Deep Residual Learning for Image Recognition, CVPR 2016](https://arxiv.org/abs/1512.03385)
- [Identity Mappings in Deep Residual Networks, ECCV 2016](https://arxiv.org/abs/1603.05027)
