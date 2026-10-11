# 03 GEMM

状态：课程理论 v1 已提供；完整 CUDA 实现和 GPU 性能待验证。

1. [分块与数据复用](01-tiling.md)：从一个输出到一个输出块，手算读取量，理解边界与同步。
2. [Tensor Core 与流水线](02-tensor-core-and-pipeline.md)：理解共同计算、缓冲区状态与调度选择。
3. [线程与矩阵元素](03-thread-layout.md)：区分存储、搬运和计算分工。
4. [低精度与缩放](04-low-precision.md)：区分量化误差、还原系数和累加方式。
5. [调度与输出](05-scheduling-and-output.md)：任务覆盖、split-K、bias 与激活的位置。
6. [DeepGEMM 固定版本阅读](../../case_studies/deepgemm/README.md)：回到源码定位等待与缩放。

完成标准：画出一个输出块需要哪些输入，计算一轮数据量，解释缓冲区何时可覆盖。无需 GPU。

运行 `python3 tools/check_theory.py` 与 `python3 tools/check_patterns.py`，检查数学与索引例子；它们不模拟 GPU 硬件。

选择分块前完成 [硬件资源预算](../../docs/hardware/resource-budget.md)，同时检查共享内存、寄存器和网格块数。

架构进阶：[Hopper 交接](../../docs/hardware/hopper-pipeline.md) → [SM100 矩阵计算](../../docs/hardware/blackwell-matmul.md) → [固定源码逐段阅读](../../case_studies/architecture-walkthrough.md)。调度练习运行 `python3 tools/check_launch_and_schedule.py`。
