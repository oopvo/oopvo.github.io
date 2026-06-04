---
layout: glossary
title: GRPO
categories: [训练方法]
related_posts:
  - DeepSeek-V3 技术报告精读：671B 参数的 MoE 奇迹
---

**Group Relative Policy Optimization**（分组相对策略优化）是 DeepSeek 自研的强化学习算法。

与传统的 PPO 不同，GRPO 不需要 Critic 模型（价值函数模型），而是通过对同一 prompt 生成的多个 response 进行分组比较来估计优势函数，大幅简化了 RL 训练流程。
