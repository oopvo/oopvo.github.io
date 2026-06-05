---
layout: glossary
title: Positional Encoding
categories: [模型架构]
---

**Positional Encoding**（位置编码）解决 Transformer 的一个固有问题：自注意力机制本身是**置换不变的**——它无法区分 "A→B" 和 "B→A"。

## 常用方法

### 正弦波位置编码（Sinusoidal）

使用不同频率的正弦和余弦函数：

```
PE(pos, 2i)   = sin(pos / 10000^(2i/d_model))
PE(pos, 2i+1) = cos(pos / 10000^(2i/d_model))
```

- **绝对位置**：每个位置有唯一的编码向量
- **相对位置**：相邻位置的编码差异体现相对距离
- **无需训练**：公式固定，可推广到任意长度

### 可学习位置编码

将位置编码作为可训练的参数，让模型自行学习位置表示。

### RoPE（旋转位置编码）

DeepSeek-V3 的 {% include gloss.html term="MLA" %} 使用的就是 RoPE，将位置信息编码到旋转矩阵中，具有良好的相对位置感知能力。
