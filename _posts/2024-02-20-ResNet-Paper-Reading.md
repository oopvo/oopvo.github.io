---
layout: post
title: 论文精读 — Deep Residual Learning for Image Recognition
categories: paper-reading
tags: [ResNet, CV, 残差网络]
---

## 基本信息

- **标题**: Deep Residual Learning for Image Recognition
- **作者**: Kaiming He, Xiangyu Zhang, Shaoqing Ren, Jian Sun
- **发表**: CVPR 2016 / TPAMI 2016（**Best Paper**）
- **引用**: 200,000+

---

## 核心问题：{% include gloss.html term="Degradation Problem" %}

随着网络深度的增加，模型的性能出现**饱和甚至下降**。这不是过拟合，而是优化困难——作者称之为 **{% include gloss.html term="Degradation Problem" %}（退化问题）**。

```
传统 CNN 深度增加时:

  20 层 CNN:  训练误差 = 0.1,  验证误差 = 0.15
  56 层 CNN:  训练误差 = 0.25, 验证误差 = 0.28  ← 更差了！

  注意：56 层的训练误差反而更高 → 不是过拟合
  原因：深层网络难以优化 → 优化困难
```

{% include gloss.html term="ResNet" %} 用了一个极其简洁但又极其有效的方法解决了这个问题。

---

## 核心思想：{% include gloss.html term="Residual Learning" %}

{% include gloss.html term="Residual Learning" %} 的思想：**不再直接学习目标映射，而是学习残差**。

### 数学变换

**传统网络**：直接学习目标映射

$$ H(x) = \text{目标映射} $$

**残差网络**：学习残差，原始映射通过加法恢复

$$ F(x) = H(x) - x \quad \Longrightarrow \quad H(x) = F(x) + x $$

```
传统: H(x) = 目标映射
残差: F(x) = H(x) - x,  H(x) = F(x) + x
```

### 为什么有效

当网络层已经足够好、不需要变化时：
- 传统网络：需要学习 **H(x) = x**（恒等映射）→ 对 SGD 来说很难
- {% include gloss.html term="ResNet" %}：只需学习 **F(x) = 0** → 把权重趋向于 0 即可

```
恒等映射:
  传统: 需要将权重收敛到单位矩阵附近 → 困难
  残差: 只需将权重收敛到 0 → 容易

类比：与其从头学做一道菜，不如学如何把半成品做得更好吃
```

---

## 实现方式：{% include gloss.html term="Shortcut Connections" %}

{% include gloss.html term="Shortcut Connections" %}（快捷连接）是实现 {% include gloss.html term="Residual Learning" %} 的具体方式：

### 残差块

```python
class BasicBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, 3, padding=1)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(out_channels, out_channels, 3, padding=1)
        self.bn2 = nn.BatchNorm2d(out_channels)

    def forward(self, x):
        identity = x                         # 保存输入

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)

        out += identity                      # Shortcut Connection！
        return self.relu(out)
```

### 可视化

```
输入 x ───→ [Conv] → [ReLU] → [Conv] ──→ F(x) ──→ ⊕ ──→ ReLU ──→ 输出
          │                                            ▲
          └────────────────────────────────────────────┘
                        恒等连接（identity mapping）

梯度反向传播:
  损失 ←── [Conv] ←── [ReLU] ←── [Conv] ←── ⊕ ←── 梯度直接流过！
                                              ▲
                                              └── 梯度走快捷连接直达
```

{% include gloss.html term="Shortcut Connections" %} 让梯度可以**直接回传到浅层**，这是 {% include gloss.html term="ResNet" %} 能训练超深网络的本质原因。

---

## 网络架构

| 层名 | 输出尺寸 | 18 层 | 34 层 | 50 层 | 101 层 | 152 层 |
|------|---------|-------|-------|-------|--------|--------|
| conv1 | 112×112 | 7×7, 64, stride 2 | ← 同左 | ← 同左 | ← 同左 | ← 同左 |
| conv2_x | 56×56 | [3×3, 64]×2 | [3×3, 64]×3 | [1×1,64; 3×3,64; 1×1,256]×3 | ×3 | ×3 |
| conv3_x | 28×28 | [3×3, 128]×2 | [3×3, 128]×4 | [1×1,128; 3×3,128; 1×1,512]×4 | ×4 | ×8 |
| conv4_x | 14×14 | [3×3, 256]×2 | [3×3, 256]×6 | [1×1,256; 3×3,256; 1×1,1024]×6 | ×23 | ×36 |
| conv5_x | 7×7 | [3×3, 512]×2 | [3×3, 512]×3 | [1×1,512; 3×3,512; 1×1,2048]×3 | ×3 | ×3 |
| FLOPs | — | 1.8G | 3.6G | 3.8G | 7.6G | 11.3G |

---

## 实验结果

{% include gloss.html term="ResNet" %} 在 ImageNet 上的表现：

| 网络 | top-1 错误率 | top-5 错误率 |
|------|------------|------------|
| VGG-16 | 28.5% | 9.3% |
| ResNet-34 | 26.7% | 8.5% |
| ResNet-50 | 24.6% | 7.5% |
| ResNet-101 | 23.6% | 7.1% |
| ResNet-152 | **23.0%** | **6.7%** |

**152 层** {% include gloss.html term="ResNet" %} 的 top-5 错误率仅 **6.7%**，以 **3.57%** 的 top-5 错误率赢得 **ILSVRC 2015 冠军**。

---

## 对学习者的启示

1. **{% include gloss.html term="Degradation Problem" %}** 提醒我们：网络并非越深越好，优化难度会随深度增加
2. **{% include gloss.html term="Residual Learning" %}** 的思想可以迁移到很多领域——有时改学习目标比直接学更容易
3. **{% include gloss.html term="Shortcut Connections" %}** 是 {% include gloss.html term="Transformer" %} 等现代架构中的标准组件
4. 一个简单的思想改变（恒等映射 → 残差），就能催生一个里程碑

---

## 参考文献

- [Deep Residual Learning for Image Recognition, CVPR 2016](https://arxiv.org/abs/1512.03385)
- [Identity Mappings in Deep Residual Networks, ECCV 2016](https://arxiv.org/abs/1603.05027)
