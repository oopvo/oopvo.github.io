---
layout: glossary
title: ResNet
categories: [模型架构]
---

**ResNet**（Residual Network，残差网络）由 Kaiming He 等人在 2015 年提出（CVPR 2016 Best Paper），通过引入残差学习解决了深层网络的退化问题。

## 核心贡献

1. 提出残差学习框架，解决了深层网络的**退化问题**
2. 成功训练 152 层深度网络（当时 VGG 只有 19 层）
3. 以 3.57% top-5 错误率赢得 ILSVRC 2015 冠军

## 基本单元

```
传统网络:        x → [Conv] → [ReLU] → [Conv] → H(x)
                                                  ▲ 直接学习目标映射

残差网络:        x ──┬────────────────────→ ⊕ → ReLU → F(x) + x
                     │                          ▲
                     └──→ [Conv] → [ReLU] → [Conv] ┘
                                                  ▲ 学习残差 F(x) = H(x) - x
```

{% include gloss.html term="Residual Learning" %} 和 {% include gloss.html term="Shortcut Connections" %} 是 ResNet 的两大关键。
