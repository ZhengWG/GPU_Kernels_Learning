# DeepGEMM：把分块、缩放和等待放回源码

阅读版本：`057ca5964aae0879ff2e0eb71ee05a3cb0ba3df7`。已核对选定文件与关键实现，未编译或在 GPU 运行。来源记录见 [sources.json](../sources.json)。

前置：[分块](../../lessons/03-gemm/01-tiling.md)、[流水线](../../lessons/03-gemm/02-tensor-core-and-pipeline.md)、[低精度](../../lessons/03-gemm/04-low-precision.md)。本次只跟踪 SM90 的一条 FP8 路径。

## 从外向内读

| 顺序 | 固定版本入口 | 带着什么问题读 |
| --- | --- | --- |
| 1 | [test_fp8_fp4.py：test_gemm](https://github.com/deepseek-ai/DeepGEMM/blob/057ca5964aae0879ff2e0eb71ee05a3cb0ba3df7/tests/test_fp8_fp4.py#L26) | 输入是什么？参考结果与误差怎样定义？ |
| 2 | [gemm.hpp：fp8_fp4_gemm_nt](https://github.com/deepseek-ai/DeepGEMM/blob/057ca5964aae0879ff2e0eb71ee05a3cb0ba3df7/csrc/apis/gemm.hpp#L78) | 检查哪些尺寸、布局和类型？选哪条架构路径？ |
| 3 | [SM90 启动实现](https://github.com/deepseek-ai/DeepGEMM/blob/057ca5964aae0879ff2e0eb71ee05a3cb0ba3df7/csrc/jit_kernels/impls/sm90_fp8_gemm_1d2d.hpp) | 分块与编译配置怎样传给 kernel？ |
| 4 | [SM90 kernel](https://github.com/deepseek-ai/DeepGEMM/blob/057ca5964aae0879ff2e0eb71ee05a3cb0ba3df7/deep_gemm/include/deep_gemm/impls/sm90_fp8_gemm_1d2d.cuh#L48) | 搬运、等待、矩阵计算与缩放在哪里？ |

## 三个容易读错的地方

**接口中的 B 不是课堂中的 K×N 存储。** 所选 NT 入口约定 `[M,K] @ [N,K].T`。先区分数学转置与实际地址步长，不能把测试中的 B 按课堂的连续 K×N 解释。

**输入不只有低精度数值。** a、b 参数包含数值张量与缩放张量。接口先调整缩放布局，再按设备和缩放类型分派；同名数学操作可能走不同 kernel。

**数据就绪与缓冲区可覆盖各有等待。** 搬运侧先等 empty barrier，再提交 A、B 以及 A 缩放的搬运；计算侧等 full barrier。矩阵计算完成后，才能允许下一轮复用对应存储。

## 对照理论找四个位置

| 理论动作 | 固定源码位置 |
| --- | --- |
| 等待缓冲区可写 | [第 182 行附近](https://github.com/deepseek-ai/DeepGEMM/blob/057ca5964aae0879ff2e0eb71ee05a3cb0ba3df7/deep_gemm/include/deep_gemm/impls/sm90_fp8_gemm_1d2d.cuh#L182) |
| 等待搬运完成 | [第 291 行附近](https://github.com/deepseek-ai/DeepGEMM/blob/057ca5964aae0879ff2e0eb71ee05a3cb0ba3df7/deep_gemm/include/deep_gemm/impls/sm90_fp8_gemm_1d2d.cuh#L291) |
| 发出矩阵乘加并等完成 | [第 304 行附近](https://github.com/deepseek-ai/DeepGEMM/blob/057ca5964aae0879ff2e0eb71ee05a3cb0ba3df7/deep_gemm/include/deep_gemm/impls/sm90_fp8_gemm_1d2d.cuh#L304) |
| 使用对应缩放累加 | [第 329 行附近](https://github.com/deepseek-ai/DeepGEMM/blob/057ca5964aae0879ff2e0eb71ee05a3cb0ba3df7/deep_gemm/include/deep_gemm/impls/sm90_fp8_gemm_1d2d.cuh#L329) |

这里 WGMMA 指 Hopper 的一类协作矩阵乘加指令；先理解输入、输出和等待关系，再深入指令布局。

## 不需要 GPU 的完成标准

画出一个 K 分块从输入到最终累加结果的路径，标出两次等待和缩放位置；解释为什么去掉等待或把所有缩放挪到最后会影响正确性。

本页是独立讲解与源码链接，没有复制上游实现。上游许可证见 [固定版本 LICENSE](https://github.com/deepseek-ai/DeepGEMM/blob/057ca5964aae0879ff2e0eb71ee05a3cb0ba3df7/LICENSE)。

线程布局、任务循环和寄存器分配的答案示例见 [逐段阅读](../architecture-walkthrough.md)。
