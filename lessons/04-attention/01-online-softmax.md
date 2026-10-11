# 逐块 Softmax：最大值变了，旧结果怎么办

前置：[Softmax 定义](../../docs/prerequisites/03-math-for-kernels.md)。这里只讨论有限分数，屏蔽位置见下一篇。

**每次只看到一块分数，也能维护整行的归一化分母。**

## 先看完整一行

普通稳定 Softmax 使用 `m=max(x)` 和 `l=sum(exp(x-m))`，最后 `p[j]=exp(x[j]-m)/l`。

若先看第一块，得到的最大值可能在下一块被超过。此时不能直接把两块各自缩放后的和相加，因为它们减去了不同的最大值。

```text
原理示意

旧块：减 m_old 后得到一组指数值 ── 重新缩放 ─┐
                                           ├→ 同一个最大值下求和
新块：减 m_new 后计算指数值 ────────────────┘
```

## 只保留两个数

处理新块 `x_block` 时：

```text
m_new = max(m_old, max(x_block))
alpha = exp(m_old - m_new)
l_new = alpha * l_old + sum(exp(x_block - m_new))
```

`m` 是已见分数的最大值，`l` 是按这个最大值缩放后的指数和。这种逐步更新避免为了求最大值和分母反复完整扫描输入。[Online normalizer calculation for softmax](https://arxiv.org/abs/1805.02867)

<details>
<summary>为什么旧结果乘 alpha？</summary>

对任意旧分数 x：`exp(x-m_new) = exp(x-m_old)*exp(m_old-m_new)`。右边第二项对所有旧分数相同，因此可以一次缩放旧的总和。

</details>

## 手算一个整齐的例子

选择分数 `[0, ln(2), ln(4)]`，分成前两个和最后一个：

| 步骤 | 最大值 m | 指数和 l |
| --- | --- | --- |
| 第一块 `[0,ln(2)]` | ln(2) | 1/2 + 1 = 3/2 |
| 第二块 `[ln(4)]` | ln(4) | 旧和乘 1/2，再加 1：7/4 |

整行在最终最大值下的指数是 `[1/4,1/2,1]`，所以权重为 `[1/7,2/7,4/7]`。

## 分母算出来，不代表所有输出都已得到

如果要求输出整行 Softmax，通常还要保留或重新读取输入，用最终的 m、l 计算每个输出。逐块更新两个数并不能凭空保存所有权重。

Attention 的机会在于：它接下来只需要这些权重对 V 做加权求和，可以连这个总和一起维护，不必单独输出完整权重矩阵。

## 自测

1. 新块最大值更小时，旧和的缩放系数是多少？
2. 把两个块各自归一化后拼接，能得到整行 Softmax 吗？

<details>
<summary>参考答案</summary>

1. 旧最大值不变，alpha=1。
2. 不能；两个块各自的权重和都是 1，拼接后总和为 2，且不同块的相对权重丢失。

</details>

下一篇：[Attention 分块融合](02-tiled-attention.md)。
