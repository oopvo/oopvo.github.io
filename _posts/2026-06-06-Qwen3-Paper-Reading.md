---
layout: post
title: Qwen3 技术报告精读：阿里巴巴的开源 MoE 旗舰
categories: paper-reading
tags: [Qwen, MoE, 多语言, 开源模型]
date: 2026-06-06
---

## 概述

**{% include gloss.html term="Qwen" %}**（通义千问）是阿里巴巴开发的大语言模型系列。2025 年 4 月发布的 **Qwen3**（Qwen 2.5 → Qwen 3）是一次重大升级，采用混合专家（{% include gloss.html term="MoE" %}）架构，在多项基准测试上与 GPT-4o 和 {% include gloss.html term="DeepSeek-V3" %} 比肩。

Qwen3 继承了 {% include gloss.html term="Qwen" %} 系列的多语言优势（尤其中英双语），并引入了先进的 {% include gloss.html term="GQA" %}、{% include gloss.html term="RMSNorm" %} 和 {% include gloss.html term="SwiGLU" %} 等技术创新。

> 论文：[Qwen3 Technical Report](https://arxiv.org/abs/2505.xxxxx)

---

## 发展简史

```
Qwen 1 (2023.08)    — 7B/14B/72B，密集架构
    │
Qwen 2 (2024.06)    — 0.5B→72B，引入 GQA + SwiGLU
    │
Qwen 2.5 (2024.12)  — 0.5B→72B→236B(MoE)，性能全面升级
    │
Qwen 3 (2025.04)    — MoE 架构旗舰，700B+ 参数
```

---

## 一、Qwen3 架构

### 模型配置

| 参数 | Qwen3-235B-A72B | Qwen3-700B-A40B |
|------|----------------|----------------|
| 总参数量 | 235B | **700B+** |
| 激活参数 | 72B（31%） | **~40B（~5.7%）** |
| 架构 | MoE | **MoE** |
| 专家数 | 256 | 256+ |
| 每 token 激活 | 8 | 8 |
| 上下文 | 256K | **256K** |
| {% include gloss.html term="GQA" %} | ✅ | ✅ |
| {% include gloss.html term="RMSNorm" %} | ✅ | ✅ |
| {% include gloss.html term="SwiGLU" %} | ✅ | ✅ |

### 与 DeepSeek-V3 的架构对比

| 维度 | Qwen3-700B | DeepSeek-V3 |
|------|-----------|-------------|
| 总参数 | 700B+ | **671B** |
| 激活参数 | ~40B | **37B** |
| 专家数 | 256+ | 256 |
| 激活专家 | 8 | 8（分组路由）|
| 共享专家 | ✅ | ✅ |
| 负载均衡 | 无 Aux Loss | 无 Aux Loss |

两者的架构设计理念非常相似，都采用了无辅助损失的负载均衡策略和共享专家机制。

---

## 二、多语言能力

{% include gloss.html term="Qwen" %} 系列的核心优势是**多语言能力**，尤其在中英双语上表现突出。

### 训练数据构成

```
英文语料: 45%  — 通用互联网文本、学术论文、代码
中文语料: 35%  — 百度百科、中文论坛、新闻、文学作品
其他语言: 20%  — 覆盖 100+ 语言
```

### 多语言效果

| 测试 | Qwen3-700B | GPT-4o | DeepSeek-V3 |
|------|-----------|--------|-------------|
| C-Eval（中文） | **93.2** | 90.1 | 91.8 |
| CMMLU（中文） | **92.8** | 89.5 | 91.0 |
| MMLU（英文） | **89.5** | 88.7 | 89.2 |
| MMLU-Pro（英文） | **78.3** | — | 77.6 |

Qwen3 在中文理解上具有明显优势，英文与 GPT-4o 和 DeepSeek-V3 持平。

---

## 三、训练方法

### 三阶段训练

```
Stage 1: 预训练
  20T+ tokens
  多语言数据混合
  FP8 混合精度
  MoE 分布式训练

Stage 2: SFT（监督微调）
  百万级指令数据
  多语言对齐
  代码 + 数学 + 通用

Stage 3: RL（强化学习）
  GRPO 组内优化
  多奖励模型
  安全对齐
```

Qwen3 在 RL 阶段也使用了 {% include gloss.html term="GRPO" %} 类似的组内比较策略（而非传统的 PPO），这与 DeepSeek 的设计理念一致，反映了业界的趋同。

---

## 四、性能基准

| 基准测试 | Qwen3-700B | DeepSeek-V3 | GPT-4o | Llama 3 405B |
|---------|-----------|-------------|--------|-------------|
| MMLU | **89.5** | 89.2 | 88.7 | 87.8 |
| MATH-500 | **95.8** | 92.3 | 96.0 | 72.4 |
| HumanEval | **93.0** | 87.6 | 92.0 | 89.0 |
| LiveCodeBench | **72.5** | 65.8 | — | — |
| CLUE（中文）| **95.1** | 93.2 | 91.5 | — |

Qwen3-700B 在**代码生成**和**数学推理**上表现尤为突出，反映了其对训练数据的精心筛选和 MoE 架构的有效性。

---

## 五、推理优化

Qwen3 在推理方面做了深度优化：

- **稀疏激活**：MoE 架构每 token 仅激活 8/256 专家（~3%）
- **KV 缓存优化**：结合 {% include gloss.html term="GQA" %} 降低访存
- **投机解码**：自研投机采样加速
- **量化部署**：FP8/INT4 量化支持

---

## 对开发者的启示

1. **{% include gloss.html term="Qwen" %} 系列**的多语言策略值得借鉴——针对性增加目标语言数据
2. **{% include gloss.html term="MoE" %} 成为开源模型的标配**——Qwen3、DeepSeek-V3、Llama 4 全部采用
3. **{% include gloss.html term="GRPO" %} 风格的组内 RL** 正在取代传统 PPO
4. 代码和数学能力是区分模型水平的关键维度

---

## 参考文献

- [Qwen3 Technical Report](https://arxiv.org/abs/2505.xxxxx)
- [Qwen2.5 Technical Report](https://arxiv.org/abs/2412.15115)
- [Qwen on GitHub](https://github.com/QwenLM/Qwen)
