---
layout: glossary
title: ResNet
categories: [模型架构]
---

**ResNet**（Residual Network，残差网络）由 Kaiming He 等人在 2015 年提出（CVPR 2016 Best Paper），通过引入残差学习解决了深层网络的退化问题。

![ResNet 残差块示意图](https://raw.githubusercontent.com/pytorch/vision/main/docs/source/_static/img/resnet.png)

## 核心贡献

1. 提出残差学习框架，解决了深层网络的**退化问题**
2. 成功训练 152 层深度网络（当时 VGG 只有 19 层）
3. 以 3.57% top-5 错误率赢得 ILSVRC 2015 冠军

## 基本单元

```
传统网络（Plain）:
  x → [Conv] → [ReLU] → [Conv] → H(x)
  ↑ 直接学习目标映射 H(x)

残差网络（ResNet）:
  x ──┬─────────────────────→ ⊕ → ReLU → 输出
      │                       ↑
      └──→ [Conv] → [ReLU] → [Conv] ┘
            F(x) = H(x) - x

  输出 = F(x) + x

  梯度回传路径:
  损失 L
    │
  ∂L/∂x = ∂L/∂y · (1 + ∂F/∂x)
    │         ↑ 梯度直接走快捷连接！永不消失
    ▼
  浅层收到完整梯度 ✅
```

**梯度分析**：反向传播时，损失对输入的偏导包含一个常数项 1（来自快捷连接），即使 ∂F/∂x 很小（权重层梯度消失），梯度依然可以直接流回浅层。

{% include gloss.html term="Residual Learning" %} 和 {% include gloss.html term="Shortcut Connections" %} 是 ResNet 的两大关键。
