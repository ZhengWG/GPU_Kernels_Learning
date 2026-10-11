# 从线程分工读到数据交接：三条固定源码路径

**每次只选一条路径，记录谁写、谁读、何时释放。** 前置：[Hopper](../docs/hardware/hopper-pipeline.md)、[SM100](../docs/hardware/blackwell-matmul.md)、[调度与输出](../lessons/03-gemm/05-scheduling-and-output.md)。提交号及文件摘要沿用 [sources.json](sources.json)，未编译或运行上游 kernel。

## 1. DeepGEMM SM90：搬运组与计算组

固定版本 `057ca5964aae0879ff2e0eb71ee05a3cb0ba3df7`，文件 `sm90_fp8_gemm_1d2d.cuh`。先选择普通 NT GEMM、无跨块 multicast 的理解路径，再扩展模板分支。

| 片段 | 读到的事实 | 为什么要这样组织 |
| --- | --- | --- |
| [初始化与寄存器分配，130 行起](https://github.com/deepseek-ai/DeepGEMM/blob/057ca5964aae0879ff2e0eb71ee05a3cb0ba3df7/deep_gemm/include/deep_gemm/impls/sm90_fp8_gemm_1d2d.cuh#L130) | full/empty barrier 分开，初始化后处理可见性；搬运与计算侧使用不同寄存器配额 | 搬运侧少保留累加值，计算侧需要更多空间；配额不是实测占用报告 |
| [角色分支，164 行起](https://github.com/deepseek-ai/DeepGEMM/blob/057ca5964aae0879ff2e0eb71ee05a3cb0ba3df7/deep_gemm/include/deep_gemm/impls/sm90_fp8_gemm_1d2d.cuh#L164) | `warp_idx` 决定进入搬运还是计算；搬运分支中再选一个线程发起 TMA | 不是所有线程重复提交同一份搬运 |
| [行布局，218 行起](https://github.com/deepseek-ai/DeepGEMM/blob/057ca5964aae0879ff2e0eb71ee05a3cb0ba3df7/deep_gemm/include/deep_gemm/impls/sm90_fp8_gemm_1d2d.cuh#L218) | 用 warp 与 lane 推出两组行索引，输入用共享内存描述符 | 布局包含线程持有的数据位置，不能按一线程一输出理解 |
| [主计算，291 行起](https://github.com/deepseek-ai/DeepGEMM/blob/057ca5964aae0879ff2e0eb71ee05a3cb0ba3df7/deep_gemm/include/deep_gemm/impls/sm90_fp8_gemm_1d2d.cuh#L291) | 等搬运、读取本轮缩放、发出 WGMMA、等待、最后一轮使用后释放 | 输入生命周期跨越异步指令；缩放必须与对应 K 段匹配 |
| [缩放累加，329 行起](https://github.com/deepseek-ai/DeepGEMM/blob/057ca5964aae0879ff2e0eb71ee05a3cb0ba3df7/deep_gemm/include/deep_gemm/impls/sm90_fp8_gemm_1d2d.cuh#L329) | 局部累加值乘对应 A/B 缩放后加入最终结果 | 不能把不同缩放段的原始结果全部相加后统一乘一个系数 |

行索引手算：这里 `r0=warp_idx*16+lane_idx//4`，`r1=r0+8`。warp 1、lane 9 得到行 18、26。相邻四个 lane 可能共享行号，但持有不同列片段；完整列布局还由 WGMMA 形状和累加片段决定，不能仅凭 r0/r1 得出完整元素地址。

调度器在搬运和计算两侧持续取得输出块。它们必须一致地遍历任务，并保持 stage 与 phase 对齐；“每个块只算一个输出块”的教学模型在这里需要扩展。

## 2. FlashAttention Hopper：K 和 V 的最后使用时刻不同

固定版本 `94e22c906678e5483fa0e9e24d8e787bc2c0ed4c`。跟踪普通前向、不带额外 QV 项的路径，其他模板分支另行确认。

| 片段 | 读到的事实 | 对应理论 |
| --- | --- | --- |
| [搬运，750 行起](https://github.com/Dao-AILab/flash-attention/blob/94e22c906678e5483fa0e9e24d8e787bc2c0ed4c/hopper/mainloop_fwd_sm90_tma_gmma_ws.hpp#L750) | K、V 各有 pipeline；先 acquire，再选择 TMA 或其他搬运分支 | 缓冲区可写与可读是两次交接 |
| [布局，1015 行起](https://github.com/Dao-AILab/flash-attention/blob/94e22c906678e5483fa0e9e24d8e787bc2c0ed4c/hopper/mainloop_fwd_sm90_tma_gmma_ws.hpp#L1015) | 为 QK 和 PV 分别取得矩阵运算切片与输入片段 | QK 输出变成 PV 输入，布局可能需要转换 |
| [首轮，1130 行起](https://github.com/Dao-AILab/flash-attention/blob/94e22c906678e5483fa0e9e24d8e787bc2c0ed4c/hopper/mainloop_fwd_sm90_tma_gmma_ws.hpp#L1130) | 等 K、计算 QK、等矩阵完成、释放 K，再施加掩码并更新 Softmax | K 的生命周期在 QK 使用完成后可结束 |
| [重叠循环，1170 行起](https://github.com/Dao-AILab/flash-attention/blob/94e22c906678e5483fa0e9e24d8e787bc2c0ed4c/hopper/mainloop_fwd_sm90_tma_gmma_ws.hpp#L1170) | 当前 QK 与前一块 PV 交错；可看到不同 wait 以及 K/V 的不同 release | V 必须等相关 PV 完成，不随 K 一起提前释放 |

源码中的 `wait<1>` 不是“等一次”：它允许最近的部分已提交工作仍在途。这里能释放 K 的依据是提交次序与已完成组，而不是模板参数恰好为 1。要修改顺序，必须重新证明哪些操作还使用哪一个缓冲槽。

## 3. FlashAttention SM100：分数、Softmax 与输出分别交接

同一提交的 Python CuTe DSL 文件 `flash_fwd_sm100.py`，先读角色、存储和分支条件，比从大循环第一行往下猜更容易。

| 片段 | 阅读结论 |
| --- | --- |
| [角色初始化，312 行起](https://github.com/Dao-AILab/flash-attention/blob/94e22c906678e5483fa0e9e24d8e787bc2c0ed4c/flash_attn/cute/flash_fwd_sm100.py#L312) | 初始表中 warp 0–3、4–7 做 Softmax，8–11 做修正，12 做 MMA，13 做输出，14 做搬运，15 留空。后面的 q_stage、TMA、输出和动态调度分支会重分配角色，不能当成所有配置的固定表 |
| [TMEM 区域，362 行起](https://github.com/Dao-AILab/flash-attention/blob/94e22c906678e5483fa0e9e24d8e787bc2c0ed4c/flash_attn/cute/flash_fwd_sm100.py#L362) | 明确列出分数和输出区域偏移，并检查总列数；部分中间值的区域复用依赖协议 |
| [分配与释放，1344 行起](https://github.com/Dao-AILab/flash-attention/blob/94e22c906678e5483fa0e9e24d8e787bc2c0ed4c/flash_attn/cute/flash_fwd_sm100.py#L1344) | MMA 角色申请 TMEM，通过同步发布地址，调用矩阵循环，退出前再次同步并释放 |
| [双槽分支，1896 行起](https://github.com/Dao-AILab/flash-attention/blob/94e22c906678e5483fa0e9e24d8e787bc2c0ed4c/flash_attn/cute/flash_fwd_sm100.py#L1896) | `use_s_ping_pong` 条件下交错 QK/PV，轮流使用 S/P 槽；这是特定分支，不可拼接为所有路径的共同流程 |
| [TMEM 读取布局，2236 行起](https://github.com/Dao-AILab/flash-attention/blob/94e22c906678e5483fa0e9e24d8e787bc2c0ed4c/flash_attn/cute/flash_fwd_sm100.py#L2236) | 选择专用加载操作，再按线程取得切片；Softmax 用的数据不是任意直接读取的一行数组 |

从这些片段能确认分工和交接结构，不能仅凭阅读证明所有分支的数值正确性或性能。没有执行和编译过的配置仍标记“未验证”。

## 三道阅读练习

1. DeepGEMM 的 warp 1、lane 9 为什么不能只负责一个输出元素？
2. Hopper Attention 中 QK 完成，是否说明前一块 PV 也完成了？
3. SM100 的初始 warp 14 角色，是否适用于所有 q_stage 配置？

<details>
<summary>答案要点</summary>

1. 行索引已有 18、26 两组，具体列还由指令片段布局决定；一个线程持有多个值。
2. 不能。它们可以属于不同在途组，必须检查提交顺序、等待语义以及相应 K/V 槽。
3. 不能。初始化后有角色重分配分支；先固定参数再画最终角色表。

</details>

三份源码均只链接与解释，不复制上游实现；许可链接见 [DeepGEMM](deepgemm/README.md) 与 [FlashAttention](flash_attention/README.md)。
