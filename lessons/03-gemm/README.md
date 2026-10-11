# 03 GEMM

状态：理论初版已提供；完整 CUDA 实现和 GPU 性能待验证。

1. [分块与数据复用](01-tiling.md)：从一个输出到一个输出块，手算读取量，理解边界与同步。
2. [Tensor Core 与流水线](02-tensor-core-and-pipeline.md)：理解共同计算、缓冲区状态与不同尺寸的调度选择。

完成标准：画出一个输出块需要哪些输入，计算一轮数据量，解释缓冲区何时可覆盖。无需 GPU。

可运行 [纯 Python 推导检查](../../tools/check_theory.py)，从仓库根目录执行 `python3 tools/check_theory.py`。它验证数学结果，不模拟 GPU 硬件。
