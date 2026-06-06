---
layout: glossary
title: PPO
categories: [训练方法]
related_posts:
  - DeepSeek-V4 技术报告精读：迈向通用人工智能的架构革命
---

**Proximal Policy Optimization**（近端策略优化）是 OpenAI 于 2017 年提出的经典强化学习算法。

## 核心思想

PPO 通过**裁剪策略更新幅度**来保证训练稳定性。每次更新时，新策略不能偏离旧策略太远，否则裁剪机制会阻止更新。

## 在 LLM 中的应用

在大模型的 RLHF 阶段，PPO 是标准的强化学习算法：

```
PPO 流程:

Actor（策略模型）──▶ 生成回答 ──▶ Reward Model ──▶ 评分
    │                                 ▲
    └────── Critic（价值模型）──────────┘
    需要额外维护 Critic 模型 → 显存翻倍
```

## PPO 的局限

- 需同时维护 Actor 和 Critic 两个模型，显存开销大
- Critic 模型训练不稳定，容易不收敛
- 超参数敏感，调参困难

{% include gloss.html term="GRPO" %} 正是为解决 PPO 的这些局限而提出。
