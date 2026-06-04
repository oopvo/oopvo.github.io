---
layout: post
title: 论文精读 — Deep Residual Learning for Image Recognition
categories: paper-reading
tags: [ResNet, CV, 残差网络]
---

## 基本信息

- **标题**: Deep Residual Learning for Image Recognition
- **作者**: Kaiming He, Xiangyu Zhang, Shaoqing Ren, Jian Sun
- **发表**: CVPR 2016 / TPAMI 2016
- **引用**: 100000+

## 核心问题

随着网络深度的增加，模型的性能出现饱和甚至下降，这并非过拟合所致，而是由于优化困难。作者称之为"退化问题"（Degradation Problem）。

## 核心思想

引入**残差学习**（Residual Learning）：不再直接学习目标映射 $H(x)$，而是学习残差 $F(x) = H(x) - x$，原始映射变为 $F(x) + x$。

这通过"快捷连接"（Shortcut Connections）实现，在不增加额外参数和计算量的情况下，让网络能够轻松学习恒等映射。

## 关键贡献

1. 提出残差学习框架，解决了深层网络的退化问题
2. 构建了 152 层的 ResNet，在 ImageNet 上取得 SOTA
3. 以 3.57% 的 top-5 错误率赢得 ILSVRC 2015 冠军
