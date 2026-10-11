# FlashAttention：在源码里找回 m、l、u

阅读版本：`94e22c906678e5483fa0e9e24d8e787bc2c0ed4c`。已核对选定文件；未构建或 GPU 验证。前置：[逐块 Softmax](../../lessons/04-attention/01-online-softmax.md)与[分块 Attention](../../lessons/04-attention/02-tiled-attention.md)。

## 先读较短的 C++ Softmax

入口：[softmax.h 的 softmax_rescale_o](https://github.com/Dao-AILab/flash-attention/blob/94e22c906678e5483fa0e9e24d8e787bc2c0ed4c/csrc/flash_attn/src/softmax.h#L137)。

| 课堂记号 | 源码中的对应值 | 看什么 |
| --- | --- | --- |
| m | `row_max` | 第一次初始化与后续更新 |
| l | `row_sum` | 旧分母如何随最大值变化而缩放 |
| u | `acc_o` | 输出累加值如何使用相同系数修正 |
| alpha | `scores_scale` | 新旧最大值之差如何进入指数 |

`Is_first` 分开处理首块。后续块更新最大值后，会同时缩放 `row_sum` 和 `acc_o`；最终在 [normalize_softmax_lse](https://github.com/Dao-AILab/flash-attention/blob/94e22c906678e5483fa0e9e24d8e787bc2c0ed4c/csrc/flash_attn/src/softmax.h#L170) 中归一化。这直接对应课堂的“分子分母使用同一缩放标准”。

## exp2 为什么没有改变数学目标

课堂用 `exp(x)`，源码使用 `exp2` 并带缩放常数。因为 `exp(x)=2^(x*log2(e))`，公式可以换底；还要核对 `1/sqrt(D)` 等缩放是否已经乘入。实际指令近似与舍入仍需数值验证，不能凭恒等式宣称逐位一致。

## 再回到主循环

[flash_fwd_kernel.h](https://github.com/Dao-AILab/flash-attention/blob/94e22c906678e5483fa0e9e24d8e787bc2c0ed4c/csrc/flash_attn/src/flash_fwd_kernel.h#L350) 调用 Softmax 更新，再继续加权 V 的计算，最后统一归一化。阅读时沿着 `acc_s` 和 `acc_o` 追踪：一个装分数，一个装输出累加值。

先选择无 dropout 的普通前向分支，理解后再读 causal、局部窗口和拆分计算，不要把所有模板分支混成一条执行流程。

## 进阶分支

| 路径 | 固定源码 | 阅读重点 |
| --- | --- | --- |
| Hopper C++ | [mainloop_fwd_sm90_tma_gmma_ws.hpp](https://github.com/Dao-AILab/flash-attention/blob/94e22c906678e5483fa0e9e24d8e787bc2c0ed4c/hopper/mainloop_fwd_sm90_tma_gmma_ws.hpp) | 搬运和矩阵计算的等待关系 |
| CuTe DSL | [softmax.py](https://github.com/Dao-AILab/flash-attention/blob/94e22c906678e5483fa0e9e24d8e787bc2c0ed4c/flash_attn/cute/softmax.py#L253) | `finalize`、`rescale_O` 与课堂公式的对应 |
| B 系列相关实现 | [flash_fwd_sm100.py](https://github.com/Dao-AILab/flash-attention/blob/94e22c906678e5483fa0e9e24d8e787bc2c0ed4c/flash_attn/cute/flash_fwd_sm100.py) | 不同参与者如何交接中间数据 |
| DSL 入口 | [interface.py](https://github.com/Dao-AILab/flash-attention/blob/94e22c906678e5483fa0e9e24d8e787bc2c0ed4c/flash_attn/cute/interface.py) | 输入限制与实现选择 |

这些入口帮助后续深入，不表示本页已逐行解析所有架构实现。

## 自测

找到源码中同时修正分母和输出的两行，再解释为何分母的跨线程合并可以安排到最终归一化阶段。答案应涉及“什么时候需要完整分母”，而不是只翻译变量名。

<details>
<summary>自测答案</summary>

`softmax_rescale_o` 中 `row_sum` 与 `acc_o_rowcol` 都乘 `scores_scale`。中间轮次先保留各线程的局部分母即可，只要它们使用一致的行最大值和缩放标准；最终除法前，`normalize_softmax_lse` 用 `quad_allreduce_` 合并分母。不能省略最终合并。

</details>

Hopper 与 SM100 的线程分工和交接详解见 [架构逐段阅读](../architecture-walkthrough.md)。

许可证见 [仓库 LICENSE](https://github.com/Dao-AILab/flash-attention/blob/94e22c906678e5483fa0e9e24d8e787bc2c0ed4c/LICENSE) 与 [CuTe 目录 LICENSE](https://github.com/Dao-AILab/flash-attention/blob/94e22c906678e5483fa0e9e24d8e787bc2c0ed4c/flash_attn/cute/LICENSE)。本页仅讲解并链接。
