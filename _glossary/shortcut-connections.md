---
layout: glossary
title: Shortcut Connections
categories: [模型架构]
---

**Shortcut Connections**（快捷连接/跳跃连接）是 ResNet 实现残差学习的核心组件。

## 工作原理

快捷连接将输入直接跳过一些层传递到后面的层：

```
输入 x ───→ [权重层] ──→ [权重层] ──→ F(x) ──→ ⊕ ──→ ReLU ──→ 输出
          │                                            ▲
          └────────────────────────────────────────────┘
                        快捷连接（恒等映射）
```

## 关键特性

- **恒等映射**：快捷连接本身不包含参数，直接传递 x
- **零额外成本**：不增加参数量和计算量
- **梯度高速公路**：梯度可以直接通过快捷连接回传到浅层，避免梯度消失

## 实现

在 PyTorch 中：
```python
def forward(self, x):
    identity = x
    out = self.conv1(x)
    out = self.relu(out)
    out = self.conv2(out)
    out += identity  # 快捷连接
    return self.relu(out)
```
