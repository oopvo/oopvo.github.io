---
layout: glossary
title: Auxiliary Loss
categories: [训练方法]
related_posts:
  - DeepSeek-V3 技术报告精读：671B 参数的 MoE 奇迹
---

**辅助损失函数**是传统 MoE 训练中用于平衡各专家负载的额外损失项。

训练时需要在主任务 Loss 上加上 α × AuxLoss，α 是超参数需要手动调节。DeepSeek-V3 提出了**无辅助损失的动态负载均衡策略**，完全去除了 α 调参的麻烦。
