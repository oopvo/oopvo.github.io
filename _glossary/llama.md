---
layout: glossary
title: Llama
categories: [模型架构]
related_posts:
  - Llama 系列技术报告精读：Meta 的开源大模型之路
---

**Llama**（Large Language Model Meta AI）是 Meta 开发的开源大语言模型系列，包括 Llama 1（2023.02）、Llama 2（2023.07）、Llama 3（2024.04）和 Llama 4（2025.04）。

## 关键技术创新

- **{% include gloss.html term="GQA" %}** — Llama 2 70B 起引入分组查询注意力，降低 KV 缓存
- **{% include gloss.html term="SwiGLU" %}** — Llama 全系列使用的激活函数
- **{% include gloss.html term="RMSNorm" %}** — 替代 LayerNorm 的归一化层
- **{% include gloss.html term="RoPE" %}** — 旋转位置编码

## Llama 3 405B 亮点

| 指标 | 数值 |
|------|------|
| 参数量 | **405B**（当时最大开源密集模型）|
| 训练数据 | **15.6T tokens** |
| GPU | 30.8M H100 GPU 小时 |
| 上下文窗口 | 128K |
| 训练方法 | SFT + PPO + DPO |
