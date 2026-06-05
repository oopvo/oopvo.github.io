---
layout: glossary
title: Self-Play RL
categories: [训练方法]
related_posts:
  - DeepSeek-V4 技术报告精读：迈向通用人工智能的架构革命
---

**自博弈强化学习**（Self-Play Reinforcement Learning）是 DeepSeek-V4 的核心训练方法。

## 原理

模型通过自我对弈生成训练数据 → 训练 → 评估的循环不断提升能力，无需外部标注：

```
  回合 1: 模型 A 生成解题过程
          模型 B（相同模型）验证并评分
          
  回合 2: 根据得分筛选高质量数据
          在高质量数据上继续训练
          
  回合 n: 模型能力持续提升
          生成的解题过程越来越精确
```

## DeepSeek-V4 的扩展

V4 将 Self-Play 扩展到三个全新领域：

- **数学证明**：模型自行生成证明步骤并验证逻辑链
- **代码合成**：生成代码 → 运行测试 → 根据结果改进
- **科学发现**：提出假设 → 设计验证 → 分析结果

## 与 GRPO 的关系

Self-Play RL 建立在 {% include gloss.html term="GRPO" %} 的基础上，GRPO 提供组内比较的奖励信号，Self-Play 提供数据生成的自动化流程。
