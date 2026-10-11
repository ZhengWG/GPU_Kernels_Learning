# 资料导航：带着问题阅读

本页资料访问日期：2026-10-11。以下是已查阅的背景来源，不表示已构建或复现对应仓库。官方在线文档和 GitHub 主分支会更新；落地代码案例时另行固定 commit 和环境。

## 官方资料

| 资料 | 先读什么 | 配合本仓库 |
| --- | --- | --- |
| [CUDA Programming Model](https://docs.nvidia.com/cuda/cuda-programming-guide/01-introduction/programming-model.html) | CPU/GPU、block、warp、SM 与内存 | 线程与执行 |
| [Intro to CUDA C++](https://docs.nvidia.com/cuda/cuda-programming-guide/02-basics/intro-to-cuda-cpp.html) | kernel 定义、启动、索引和边界 | 从 Python 到 C++ |
| [Writing SIMT Kernels](https://docs.nvidia.com/cuda/cuda-programming-guide/02-basics/writing-cuda-kernels.html) | 内存空间、共享数据与同步 | 存储、同步 |
| [Asynchronous Execution](https://docs.nvidia.com/cuda/cuda-programming-guide/02-basics/asynchronous-execution.html) | stream、event、CPU 等待 | 同步与计时 |
| [Best Practices Guide](https://docs.nvidia.com/cuda/cuda-c-best-practices-guide/index.html) | 合并访存、共享内存、occupancy | 转置和 GEMM 前 |
| [Floating Point](https://docs.nvidia.com/cuda/floating-point/index.html) | 运算顺序、舍入与乘加 | 正确性测试 |
| [NVCC](https://docs.nvidia.com/cuda/cuda-programming-guide/02-basics/nvcc.html) | 编译流程与目标架构 | 软件栈 |
| [Compute Sanitizer](https://docs.nvidia.com/compute-sanitizer/ComputeSanitizer/index.html) | memcheck、racecheck、synccheck | 调试错误 |
| [Nsight Compute](https://docs.nvidia.com/nsight-compute/ProfilingGuide/index.html) | Roofline、内存图与采集开销 | 性能判断 |
| [Hopper Tuning](https://docs.nvidia.com/cuda/hopper-tuning-guide/index.html) | TMA、资源限制与块间协作 | H20 进阶背景 |
| [Blackwell Tuning](https://docs.nvidia.com/cuda/blackwell-tuning-guide/index.html) | 型号差异、资源与兼容性 | B 系列扩展 |
| [PyTorch Tensor Views](https://docs.pytorch.org/docs/main/tensor_view.html) | transpose、contiguous、reshape | 张量与内存布局 |

## GitHub 代码资料

| 项目 | 用途 | 建议阅读阶段 |
| --- | --- | --- |
| [NVIDIA/cuda-samples](https://github.com/NVIDIA/cuda-samples) | 设备查询、基础计算、内存与工具示例 | 第一个 CUDA 实验起；先选一个示例，不整库通读 |
| [siboehm/SGEMM_CUDA](https://github.com/siboehm/SGEMM_CUDA) | 从直接矩阵乘法到逐步优化的代码 | shared-memory GEMM 之后 |
| [karpathy/llm.c](https://github.com/karpathy/llm.c) | C/CUDA 模型训练与算子实现 | 会归约与 GEMM 后，按单个算子阅读 |
| [NVIDIA/cutlass](https://github.com/NVIDIA/cutlass) | 矩阵计算、数据布局、C++ 与 DSL 实现 | 手写分块和 Tensor Core 入门之后 |

原有的 CUDA 120 Days、KDA、SOLAR、KDA-Pilot、FlashAttention 和 DeepGEMM 继续保留在 [首页](../README.md)。这里增加的是它们之前所需的基础入口。

## 阅读方法

每次只回答三个问题：输入如何存储？线程如何分工？中间数据在哪里复用？先画出一个小尺寸例子，再读指令和模板。上游性能图仅说明其记录的设备和环境，不作为本仓库的成绩。
