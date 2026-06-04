---
layout: glossary
title: DSA
categories: [模型架构]
related_posts:
  - DeepSeek-V3 技术报告精读：671B 参数的 MoE 奇迹
---

**DeepSeek Sparse Attention**（DeepSeek 稀疏注意力）是 V3.2 引入的两级注意力机制。

- **闪电索引器** — 快速筛选出与当前 token 最相关的 k 个 token
- **细粒度 token 选择** — 对选中的 token 进行标准注意力计算

将注意力复杂度从 O(n²) 降至 **~O(n·k)**，其中 k=2048（远小于序列长度 n）。
