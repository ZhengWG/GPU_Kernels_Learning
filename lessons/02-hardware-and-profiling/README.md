# 02 硬件与测量

状态：背景讲解已有初版，配套 GPU 微实验待实现。

阅读 [硬件参数含义](../../docs/hardware/README.md)、[存储与数据访问](../../docs/hardware/memory.md)、[性能判断](../../docs/methodology/performance-basics.md)。

练习：画出转置前后的访存地址，解释列读取为什么可能发生共享内存冲突，手算向量相加的数据量。之后用 H20 实测验证假设，记录不符合预期的情况。

无 GPU 时可继续阅读 [经典算子的计算方式](operator-patterns.md)，完成手算与自测。

继续阅读 [归约与前缀和](reduction-and-scan.md)、[RoPE 成对旋转](rope.md)。

## 把原理展开成可验证的步骤

1. [资源预算](../../docs/hardware/resource-budget.md)：比较三个 GEMM 分块候选，先判断空间与并行机会。
2. [矩阵转置](transpose.md)：从直接转置到共享块、加一列，分别检查读写边界。
3. [LayerNorm 与 RMSNorm](normalization.md)：从两遍参考到分组统计，比较保留与重读输入。
4. [融合](fusion.md)：数清中间数组读写，明确输出要求和舍入位置。

从仓库根目录运行 `python3 tools/check_resource_budget.py` 和 `python3 tools/check_classic_operators.py`。完成标准：复算章节表格，解释每一步优化的收益与代价；CPU 检查通过不代表 GPU 实验完成。
