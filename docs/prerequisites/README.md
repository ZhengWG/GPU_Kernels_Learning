# 必要背景

默认会基本 Python，不要求会 C++。下面已经提供背景讲解、原理图和自测题；代码片段用于解释概念，完整 GPU 实验另行提供。

## 阅读顺序

```mermaid
flowchart LR
    A[1. 指针与 C++ 语法] --> B[2. 张量在内存中怎么放]
    B --> C[3. 矩阵乘法与 Attention]
    C --> D[4. 浮点误差]
    D --> E[5. GPU 线程怎么分工]
    E --> F[6. 数据在哪里]
    F --> G[7. 同步在等谁]
    G --> H[8. 性能怎么判断]
```

原理示意：这是推荐顺序，不是运行流程。

| 章节 | 读完能回答什么 | 何时需要 |
| --- | --- | --- |
| [从 Python 到 C++](01-python-to-cpp.md) | 指针是什么？数组长度在哪里？GPU 函数如何启动？ | 第一个 CUDA 实验前 |
| [张量与内存布局](02-tensors-and-layout.md) | shape 和 stride 有什么区别？转置一定拷贝吗？ | 所有算子 |
| [计算所需的数学](03-math-for-kernels.md) | GEMM 的三个维度是什么？Softmax 沿哪一维计算？ | GEMM / Attention 前 |
| [浮点数与误差](04-floating-point.md) | 为什么两份正确实现也可能结果不同？ | 写正确性测试前 |
| [线程与 GPU 执行](../hardware/execution.md) | thread、warp、block、SM 如何对应？ | CUDA 入门 |
| [存储与数据访问](../hardware/memory.md) | 连续读取为什么有帮助？共享内存何时有用？ | 转置与 GEMM 前 |
| [同步与异步](05-synchronization.md) | 线程、CPU 和不同任务分别如何等待？ | 归约与计时前 |
| [性能判断](../methodology/performance-basics.md) | 慢在搬运、计算还是启动？ | 首次优化前 |
| [软件与编译流程](../software-stack/README.md) | 驱动、Toolkit、PyTorch 和编译器各做什么？ | 配环境与读源码时 |

先学前四篇，再结合第一个 CUDA 实验读后五篇，不要求一次记住所有概念。Hopper 的异步搬运等进阶知识在 [硬件入口](../hardware/README.md) 中继续学习。

原始资料的选读范围见 [资料导航](../references.md)。返回 [项目首页](../../README.md)。
