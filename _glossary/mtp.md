---
layout: glossary
title: MTP
categories: [训练方法]
related_posts:
  - DeepSeek-V3 技术报告精读：671B 参数的 MoE 奇迹
---

**Multi-Token Prediction**（多 Token 预测）是 DeepSeek-V3 在训练时采用的创新目标函数。

传统语言模型训练时每个位置只预测**下一个** token，而 MTP 让模型同时预测**多个未来 token**，强制学习更长距离的依赖关系。

MTP 在推理时**无额外开销**（只取第一步预测），但在代码生成和数学推理任务上提升显著。
