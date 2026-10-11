# KDA / KDA-Pilot：先定义什么才算优化成功

固定版本：KDA `ef6ce617693ef0782b3ecb9f37e39bbf10226a90`；KDA-Pilot `bac966ba09fe22f6220ff527c7c3555e539d5fe6`。本页只阅读流程与一个实验的测量约定，未运行代理或 GPU benchmark。

## 从任务定义开始

[KDA agent-flow](https://github.com/NVlabs/kda/blob/ef6ce617693ef0782b3ecb9f37e39bbf10226a90/docs/agent-flow.md#L23) 先列输入输出、正确性要求、约束、验证方式与接受条件。学习时先写一张表：

| 问题 | 以矩阵转置为例 |
| --- | --- |
| 数学语义 | B[j,i]=A[i,j] |
| 输入范围 | 指定 dtype、形状、步长 |
| 必须保留什么 | 每个元素的值和指定的输出布局 |
| 如何发现漏写 | 输出先填特殊值，再检查全部位置 |
| 如何比较性能 | 基线和候选使用同一计时范围 |

没有 GPU 时，最后一项写“待测”，其余仍可以完整定义。

## 一个真实案例的测量口径

阅读 KDA-Pilot 中 attention concat/copy 任务的 [benchmark_method.md](https://github.com/BBuf/KDA-Pilot/blob/bac966ba09fe22f6220ff527c7c3555e539d5fe6/diffusion/kernels/b200_diffusion_attention_concat_copy__multi_model/docs/benchmark_method.md)。

这个任务以 CUDA Events 为主要 GPU 计时，预分配输出与必要缓冲，输入构造和编译不计入计时；先验证正确性，再比较基线与候选。不要把仓库首页其他任务的端到端口径套过来。

这是数据拷贝任务，要求逐位保留值，不能借用 GEMM 的浮点容差。阅读 [correctness.py](https://github.com/BBuf/KDA-Pilot/blob/bac966ba09fe22f6220ff527c7c3555e539d5fe6/diffusion/kernels/b200_diffusion_attention_concat_copy__multi_model/bench/correctness.py)，寻找输出预填值和独立参考检查的对应实现。

## 报告要记录退步

| 候选 | 改了什么 | 正确性 | 性能 | 结论 |
| --- | --- | --- | --- | --- |
| 教学假设 A | 减少一次中间复制 | 尚待验证 | 尚待测量 | 不能宣称更快 |

同一个实现与自身比较应接近 1 倍，这可检查测量工具是否偏向某一路径。真实输入中变慢的尺寸也要保留，不能只展示最好的一行。

KDA 的许可和第三方材料约定见其 [LICENSE](https://github.com/NVlabs/kda/blob/ef6ce617693ef0782b3ecb9f37e39bbf10226a90/LICENSE)。这里只链接上游材料，后续引入代码时需逐文件核对适用条款。
