---
layout: glossary
title: Degradation Problem
categories: [模型架构]
---

**Degradation Problem**（退化问题）是指随着网络深度增加，模型性能出现饱和甚至下降的现象。

## 与过拟合的区别

| 特性 | 过拟合 | 退化问题 |
|------|--------|---------|
| 训练误差 | 很低 | **较高** |
| 验证误差 | 上升 | 下降 |
| 原因 | 模型过度拟合噪声 | **优化困难** |

退化问题的本质是：**深层网络难以学习恒等映射**。当额外层对已有表示没有贡献时，理论上应该输出恒等映射，但实际中 SGD 难以学到 F(x) = x。

## ResNet 的解决方案

ResNet 通过 {% include gloss.html term="Residual Learning" %} + {% include gloss.html term="Shortcut Connections" %} 解决了这一难题——让网络可以轻松学习 F(x) = 0 来退化为恒等映射。
