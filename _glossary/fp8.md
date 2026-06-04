---
layout: glossary
title: FP8
categories: [训练方法]
related_posts:
  - DeepSeek-V3 技术报告精读：671B 参数的 MoE 奇迹
---

**8-bit Floating Point**（8 位浮点数）是一种低精度数值格式。

相比 FP16（16 位）可减少 **50%** 显存占用，相比 FP32 减少 **75%**。DeepSeek-V3 全程使用 FP8 混合精度训练，显著降低了训练成本（仅 $5.6M）。

FP8 需要硬件支持（NVIDIA H800 等），训练时需要特殊的缩放策略防止精度丢失。
