---
layout: glossary
title: MoE
categories: [模型架构]
related_posts:
  - DeepSeek-V3 技术报告精读：671B 参数的 MoE 奇迹
---

**Mixture of Experts**（混合专家模型）是一种将模型分解为多个「专家」子网络的架构，每个 token 仅激活部分专家，在相同计算量下大幅增加模型容量。

## DeepSeek-V3 的 MoE 配置

- 总参数量 **671B**，每 token 仅激活 **37B**（5.5%）
- **256** 个路由专家 + **1** 个共享专家
- 每 token 选择 **8 个专家**（Top-8），分 8 组先选 Top-4 组
- 前 3 层使用 dense 计算保证训练稳定性

## 无 Aux Loss 负载均衡

传统 MoE 使用辅助损失平衡专家负载，DeepSeek-V3 提出**动态 bias 调整**策略：负载高的专家降低 bias → 减少被选概率，负载低的提高 bias → 增加被选概率，完全不需要辅助损失。
