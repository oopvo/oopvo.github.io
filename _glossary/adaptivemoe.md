---
layout: glossary
title: AdaptiveMoE
categories: [模型架构]
related_posts:
  - DeepSeek-V4 技术报告精读：迈向通用人工智能的架构革命
---

**自适应 MoE 路由**（Adaptive Mixture of Experts）是 DeepSeek-V4 对 {% include gloss.html term="MoE" %} 路由机制的改进。

## 核心思路

传统 {% include gloss.html term="MoE" %} 对所有 token 一视同仁，每个 token 激活固定数量的专家。AdaptiveMoE 根据**输入困难度动态调整激活专家数量**：

```
简单 token（如"的"、"是"、"和"）:
  激活 2-4 个专家 → 快速处理

普通 token（普通词汇、简单概念）:
  激活 8 个专家 → 标准处理

复杂 token（专业术语、数学符号）:
  激活 16 个专家 → 深度处理
```

## 困难度评估

使用一个轻量级**路由预测头**（仅 1 层 MLP，额外参数 < 0.01%）评估每个 token 的困难度：

```
困难度分数 = σ(MLP(hidden_state))
0.0 → 简单（4 专家）
0.5 → 普通（8 专家）
1.0 → 复杂（16 专家）
```

## 效果

| 指标 | 标准 MoE | AdaptiveMoE |
|------|---------|-------------|
| 平均激活参数 | 37B | **29B**（↓ 22%）|
| 复杂任务准确率 | baseline | **+3.2%** |
| 推理速度 | baseline | **+18%** |
