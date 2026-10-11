# 学习路线

状态：背景、GEMM 与 Attention 理论已有初版，当前没有实际 GPU 环境。优先补理论和 CPU 数学验证，完整 GPU 实验与性能验证后续开展。

先读 [背景知识入口](prerequisites/README.md)：C++、张量布局、算子数学、浮点数、线程执行、存储、同步、性能判断与编译流程。来源见 [资料导航](references.md)。

| 阶段 | 先学什么 | 有 GPU 后的完成标准 |
| --- | --- | --- |
| 00 必要背景 | Python 循环到 C++；指针、索引、张量形状和浮点误差 | 用 Python 写参考结果，解释元素在内存中的位置 |
| 01 CUDA 入门 | 线程分工、分配内存、拷贝、启动、同步与错误检查 | 完成向量相加并验证边界输入 |
| 02 硬件与测量 | 存储位置、连续读写、线程协作、计时 | 用转置与归约实验解释数据访问的影响 |
| 03 GEMM | 分块、复用、Tensor Core、搬运与计算重叠 | 逐版验证，解释收益与退化 |
| 04 Attention | 先学 Softmax，再学逐块更新和融合 | 完成前向计算，说明误差和中间存储变化 |
| 05 综合实践 | 真实输入、成熟实现、性能分析与版本记录 | 提交完整且可复现的优化报告 |

## 当前无 GPU 路线

1. 完成背景章节的图解与自测。
2. 阅读 [经典算子计算方式](../lessons/02-hardware-and-profiling/operator-patterns.md)，区分归约、前缀和、归一化和融合。
3. 阅读 [GEMM 理论](../lessons/03-gemm/README.md)，画出数据复用和缓冲区使用顺序。
4. 阅读 [Attention 理论](../lessons/04-attention/README.md)，手算逐块 Softmax 与加权输出。
5. 运行 `python3 tools/check_theory.py`，在 CPU 上验证公式与边界。
6. 阅读线程布局、低精度、RoPE、GQA 与反向章节，运行 `python3 tools/check_patterns.py` 和 `python3 tools/check_attention_backward.py`。
7. 按 [固定版本案例](../case_studies/README.md) 跟踪源码，完成 [无 GPU 综合作业](../lessons/05-capstone/README.md)。
8. 完成 [资源预算](hardware/resource-budget.md) 与 [经典算子逐步推导](../lessons/02-hardware-and-profiling/README.md)，运行两份配套 CPU 检查。
9. GPU 可用后再锁定环境、实现并测量。

CPU 检查不能发现 CUDA 读写冲突，也不验证吞吐量；理论完成与 GPU 验证分别记录。

H20 为实际验证主线。B 系列课程独立声明具体型号和验证状态。没有相应 GPU 时可以阅读原理，不能声称完成性能复现。

返回 [项目首页](../README.md)。

## 尚待展开的理论

本轮已补资源预算、转置、归一化和融合。理论部分尚未全部完成：变长/分页 KV cache、跨块 Attention 合并、掩码与 GQA 反向，以及 Hopper/B 系列具体异步指令与同步配合，仍需后续章节。
