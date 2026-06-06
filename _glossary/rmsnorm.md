---
layout: glossary
title: RMSNorm
categories: [模型架构]
---

**RMSNorm**（Root Mean Square Layer Normalization）是 LayerNorm 的简化变体，仅对输入的均方根进行归一化，省略了均值中心化步骤。

## 公式

```
LayerNorm:  y = (x - μ) / σ ⊗ γ + β
RMSNorm:    y = x / RMS(x) ⊗ γ

其中 RMS(x) = √(1/n Σ x²)
```

## 优势

- **计算高效**：省去均值计算，约节省 10-15% 归一化时间
- **效果等价**：在 Transformer 中与 LayerNorm 表现相当
- **广泛采用**：Llama、Qwen、Mistral 等主流模型均使用 RMSNorm
