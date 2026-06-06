---
layout: glossary
title: SwiGLU
categories: [模型架构]
---

**SwiGLU**（Swish-Gated Linear Unit）是 LLaMA 系列使用的激活函数，结合了 Swish 激活和门控线性单元（GLU）的优势。

## 公式

```
SwiGLU(x) = Swish(xW) ⊗ (xV)
```

其中 ⊗ 是逐元素乘法，Swish 是平滑版 ReLU。

## 与 ReLU 对比

| 特性 | ReLU | SwiGLU |
|------|------|--------|
| 表达力 | 线性 | **非线性门控** |
| 梯度 | 负数区域为零 | **处处可导** |
| 模型质量 | baseline | **+0.5-1.0%** |
| 参数 | 较少 | 多 1/3（需额外投影 V）|

SwiGLU 在现代大模型（Llama、PaLM、Gemma、Qwen）中已成为标配。
