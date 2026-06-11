---
layout: post
title: Qwen3 技术报告精读
date: 2026-03-06
categories: paper-reading
tags: [Qwen, MoE, 多语言, 开源模型, 深度分析]
date: 2026-06-06
---

> 2025 年 4 月 29 日发布，Qwen3 是阿里巴巴通义千问系列的最新成员。235B 总参数仅激活 22B（9.4%），多项基准超越 DeepSeek-R1 和 OpenAI o1，Apache 2.0 协议开源。

---

## 一、模型家族概览

{% include gloss.html term="Qwen" %} 3 提供了完整的模型家族：

| 类型 | 规模 | 特点 |
|------|------|------|
| **密集模型** | 0.6B / 1.7B / 4B / 8B / 14B / 32B | 高效率，单卡可跑 |
| **MoE 旗舰** | 235B-A22B（总 235B，激活 22B） | 性能巅峰 |
| **MoE 中型** | 30B-A3B | 性价比之选 |

**最令人震惊的数据**：Qwen3-4B（40 亿参数）性能匹敌 Qwen2.5-72B（720 亿参数）——**18 倍效率提升**！而 Qwen3-0.6B 仅需 1.49GB 内存即可本地运行。

---

## 二、{% include gloss.html term="MoE" %} 架构深度解析

### 核心配置

| 参数 | Qwen3-235B | DeepSeek-V3 | Llama 4 Maverick |
|------|-----------|-------------|------------------|
| 总参数量 | **235B** | 671B | 400B |
| 激活参数 | **22B（9.4%）** | 37B（5.5%） | 17B（4.3%） |
| 专家池 | 128（选 8） | 256（选 8+1 共享） | 128（选 1+1 共享）|
| **共享专家** | **❌ 无** | ✅ 有 | ✅ 有 |
| 注意力机制 | **GQA** | **MLA** | GQA |
| MoE 模式 | 密集层/MoE 层交替 | 全部 MoE（前 3 层 dense）| 密集层/MoE 层交替 |
| 上下文 | 32K（YaRN 可扩至 131K）| 128K | **1M** |
| 训练数据 | **36T tokens** | 14.8T | — |
| 许可证 | **Apache 2.0** | 自定义 | 自定义 |

### 关键设计决策

**1. 为什么不用共享专家？**

{% include gloss.html term="DeepSeek-V3" %} 和 Llama 4 都使用了一个"共享专家"——每个 token 都会激活它。Qwen3 团队实验发现**共享专家没有带来可衡量的收益**，于是直接去掉，简化了推理优化。

```
DeepSeek-V3 的 MoE 层:
  输入 → [共享专家（始终激活）] + [128 个路由专家中选 8 个]
         = 每 token 激活 9 个专家 ✅

Qwen3 的 MoE 层:
  输入 → [128 个路由专家中选 8 个]
         = 每 token 激活 8 个专家（更简单，效果一样）
```

**2. 为什么用 GQA 而不是 MLA？**

{% include gloss.html term="DeepSeek-V3" %} 的 {% include gloss.html term="MLA" %} 通过低秩压缩大幅减少 KV 缓存，但工程实现复杂。Qwen3 选择了更简单的 {% include gloss.html term="GQA" %}：

```
GQA（Qwen3 的选择）:
  优点：实现简单，生态成熟，与现有框架兼容
  缺点：KV 缓存比 MLA 大

MLA（DeepSeek 的选择）:
  优点：KV 缓存减少 97%，推理更高效
  缺点：需要自定义 kernel，工程复杂度高

结论：两种方案都可行，取决于团队工程能力
```

**3. 密集/MoE 交替设计**

Qwen3 在 94 个 Transformer 块中**交替使用密集层和 MoE 层**。这类似于 Llama 4 的设计，但与 DeepSeek 的全部 MoE 不同。

```
Qwen3 的 94 层结构:
  层 1: 密集 FFN → 层 2: MoE(8/128) → 层 3: 密集 → 层 4: MoE → ...

  密集层处理通用知识，MoE 层处理专业分工
  交替设计比全 MoE 更稳定，训练更容易
```

### MoE 设计模式对比（2025）

| 模型 | 专家池 | 激活策略 | 共享专家 | 设计哲学 |
|------|--------|---------|---------|---------|
| **Qwen3** | 128 | Top-8 | ❌ | 多专家专精 |
| **DeepSeek-V3** | 256 | Top-8 + 1 共享 | ✅ | 极大规模 |
| **Llama 4** | 128 | Top-1 + 1 共享 | ✅ | 少专家保守 |
| **Kimi K2** | 256+ | Top-8 + 1 共享 | ✅ | 超大容量 |

---

## 三、训练方法

### 训练数据

- **36 万亿 tokens**（DeepSeek-V3 的 2.4 倍，Llama 3 的 2.3 倍）
- 多语言混合：英语 ~60%，中文 ~25%，其他语言 ~15%
- 代码和数学数据大幅上采样
- 多阶段质量过滤

### 训练流程

```
Phase 1 — 预训练:
  36T tokens
  FP8 混合精度
  MoE 分布式训练
  连续训练（无中断）

Phase 2 — SFT（监督微调）:
  百万级指令数据
  多语言对齐
  代码 + 数学 + 通用指令
  
Phase 3 — RL（强化学习）:
  组内比较策略（类似 GRPO）
  多奖励模型
  安全对齐
```

Qwen3 在 RL 阶段也使用了类似 {% include gloss.html term="GRPO" %} 的组内比较策略（而非传统 PPO），这与 {% include gloss.html term="DeepSeek-V3" %} 的设计一致。

---

## 四、混合推理：快思考 + 慢思考

Qwen3 是首个实现**混合推理**的开源模型——同一个模型同时支持两种推理模式：

```
快速思考（Direct Response）:
  输入问题 → 直接生成答案
  适用：简单问答、事实查询
  特点：低延迟，低成本

慢速思考（Multi-step Reasoning）:
  输入问题 → 展开思考链 → 逐步推理 → 生成答案
  适用：数学证明、逻辑推理、代码调试
  特点：高精度，可控制推理深度

「思考预算」（Thinking Budget）机制:
  用户可以设置 max_thinking_tokens 来控制推理深度
  简单问题设少 → 快速回答
  复杂问题设多 → 深度推理
  同一个模型，两种模式，自由切换
```

---

## 五、性能基准

### Qwen3-235B vs DeepSeek-R1 vs Llama 4

| 基准测试 | Qwen3-235B | DeepSeek-R1 | 胜出 |
|---------|-----------|-------------|------|
| **AIME 2025**（数学奥赛）| **81.5** | 70.0 | ✅ Qwen3 |
| **AIME 2024** | **85.7** | 79.8 | ✅ Qwen3 |
| **LiveCodeBench v3**（编程）| **70.7** | 64.3 | ✅ Qwen3 |
| **Arena-Hard**（人类偏好）| **95.6** | 93.2 | ✅ Qwen3 |
| **CodeForces**（竞赛编程）| **2056 ELO** | 2029 | ✅ Qwen3 |
| **MMLU** | **~86.0** | ~84.5 | ✅ Qwen3 |

> Qwen3-235B 总参数量仅 DeepSeek-R1 的 **1/3**，但在几乎所有基准上全面超越！

### 密集模型的惊人效率

| 模型 | 参数量 | AIME 2025 | 效率比 |
|------|--------|----------|--------|
| Qwen3-32B（密集）| 32B | **72.9** | ⭐⭐⭐⭐⭐ |
| DeepSeek-R1 | ~671B | 70.0 | ⭐ |
| Qwen3-4B | 4B | 匹配 Qwen2.5-72B | **18× 提升** |

---

## 六、Qwen3 vs DeepSeek-V3：架构哲学对比

| 维度 | Qwen3 | DeepSeek-V3 |
|------|-------|-------------|
| **总参数量** | 235B（够用就好）| 671B（越大越好）|
| **激活参数** | 22B（9.4%）| 37B（5.5%）|
| **专家策略** | 无共享专家，精简 | 共享专家 + 256 专家池 |
| **注意力** | GQA（成熟稳定）| MLA（极致创新）|
| **推理模式** | 混合推理（快+慢）| 标准推理 |
| **训练数据** | 36T（更多数据）| 14.8T（更精炼）|
| **开源协议** | Apache 2.0 ✅ | 自定义 ⚠️ |
| **核心优势** | 效率、中文、混合推理 | 推理速度、长上下文 |

**一句话总结**：Qwen3 追求"用更少的参数达到更好的效果"，DeepSeek-V3 追求"用更大的模型覆盖更多场景"。两种路线各有千秋。

---

## 七、推理优化

- **稀疏激活**：MoE 架构每 token 仅激活 8/128 专家（6.25%）
- **KV 缓存优化**：结合 {% include gloss.html term="GQA" %} 降低访存
- **投机解码**：自研投机采样加速生成
- **量化部署**：FP8/INT4 量化支持
- **本地运行**：0.6B 仅需 1.49GB 内存，手机可跑

---

## 参考文献

- [Qwen3 Technical Report, arXiv 2025](https://arxiv.org/abs/2505.xxxxx)
- [Qwen2.5 Technical Report, arXiv:2412.15115](https://arxiv.org/abs/2412.15115)
- [Sebastian Raschka: The Big LLM Architecture Comparison 2025](https://sebastianraschka.com/blog/2025/the-big-llm-architecture-comparison.html)
- [FriendliAI: The Rise of MoE — Comparing 2025's Leading MoE Models](https://friendli.ai/blog/moe-models-comparison)
