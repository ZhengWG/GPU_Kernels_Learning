# Attention 反向：小变化如何传回输入

**反向计算回答：输出变化一点，哪些输入该怎样变化。** 前置：[Attention 数学](../../docs/prerequisites/03-math-for-kernels.md)和[分块前向](02-tiled-attention.md)。这里只讨论单头、无 dropout、无掩码、有限输入。

## 先把符号固定

```text
前向：Q、K → S=QK转置/sqrt(D) → P=softmax(S) → O=PV
反向：dO  → dV 与 dP → dS → dQ 与 dK
```

原理示意。`dO` 是后续计算传来的变化权重；不是再运行一次前向。这里的 d 表示对一个标量目标的导数。

## 两次矩阵乘法之间夹着 Softmax

| 步骤 | 公式 |
| --- | --- |
| O=PV | `dV = P.T @ dO`，`dP = dO @ V.T` |
| P=softmax(S) | 每行 `dS[j] = P[j]*(dP[j]-sum(P*dP))` |
| S=QK.T/sqrt(D) | `dQ=dS@K/sqrt(D)`，`dK=dS.T@Q/sqrt(D)` |

Softmax 每行输出和固定为 1，所以一个位置的变化会影响同一行其他位置。括号里的行求和项正是在处理这种联系。

<details>
<summary>一个两元素手算例子</summary>

若 P=[1/4,3/4]，dP=[2,0]，则 `sum(P*dP)=1/2`，得到 dS=[3/8,-3/8]。两项相加为零；整行分数一起加常数不会改变 Softmax，这与结果一致。

</details>

## 为什么前向没保存 P，反向仍能算

前向可以保留每行的归一化统计。反向读取 Q、K 后重新计算小块分数，用保存的统计还原小块权重，再计算梯度。它增加重算，减少保存大矩阵的需求。[FlashAttention 论文](https://arxiv.org/abs/2205.14135)

对于一行，令 `L = m + log(l)`，则权重可写为 `exp(S-L)`。这里 L 是该行统计值，与序列长度的符号无关。

不同查询块都会贡献到相同 K、V 的梯度，因此反向多了结果合并问题。共享 Q/K/V 头、causal 掩码、dropout 都要另外推导，不能照搬本页最简实现。

## 没有 GPU 怎样验证

用 Python 的小矩阵参考算出一个标量目标。把某个输入加减很小的 h，近似导数：`(f(x+h)-f(x-h))/(2h)`，再与公式比较。h 太大时近似误差大，太小时舍入误差明显。

配套 [check_attention_backward.py](../../tools/check_attention_backward.py) 使用 Python 标准库检查 Q、K、V 的各个元素。该检查验证数学推导，不验证 CUDA 同步或低精度误差。
