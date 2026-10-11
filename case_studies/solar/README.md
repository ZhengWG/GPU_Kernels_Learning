# SOLAR：把性能估算的假设写出来

阅读版本：`8f6e76c1912213e82756217e92d147aedd010f7a`。只核对源码与配置，未运行上游分析流程。前置：[性能判断](../../docs/methodology/performance-basics.md)。

## 阅读顺序

| 文件 | 先回答什么 |
| --- | --- |
| [Matmul 示例](https://github.com/NVlabs/SOLAR/blob/8f6e76c1912213e82756217e92d147aedd010f7a/examples/Matmul/Matmul.py) | 要分析的是哪种计算？ |
| [SOL_GUIDE](https://github.com/NVlabs/SOLAR/blob/8f6e76c1912213e82756217e92d147aedd010f7a/docs/SOL_GUIDE.md) | 如何定义计算量与存储量？ |
| [B200 配置](https://github.com/NVlabs/SOLAR/blob/8f6e76c1912213e82756217e92d147aedd010f7a/configs/arch/B200.yaml) | 参数单位是什么，属于哪个设备？ |
| [perf_model.py](https://github.com/NVlabs/SOLAR/blob/8f6e76c1912213e82756217e92d147aedd010f7a/solar/perf/perf_model.py#L204) | 如何从运算数、字节数算出周期？ |

## 把模型翻译成手算

所选实现分别估算矩阵计算周期和内存周期，并取较大者；不同融合假设使用不同的字节量。先跟踪 `compute_tc_cycles`、`fused_mem_cycles` 和 `fused_total_cycles`，再读整个文件。

假设计算需 100 个周期，搬运需 160 个周期，理想重叠下模型给出 160。若融合后搬运变成 80，估算变成 100，而不是 80。此处数值是假设，不是 B200 或 H20 实测。

## 模型不等于硬件事实

该固定版本在 [第 206 行附近](https://github.com/NVlabs/SOLAR/blob/8f6e76c1912213e82756217e92d147aedd010f7a/solar/perf/perf_model.py#L206) 计算其他操作的周期统计，但没有把它们加进 SOL 的计算项。这是模型选择，不能概括成“所有逐元素操作都不耗计算时间”。

还需区分 MAC 与 FLOP：一次乘加通常计为 1 MAC 或 2 FLOP，不能用前者的数量除以后者的峰值而忘记换算。融合是否可行、缓存能复用多少、真实启动和同步成本，都需要单独核查。

## 无 GPU 练习

选一个 GEMM，列出运算数、至少要读写的字节数、采用的精度和参数单位。分别计算“中间数据全部保存”与“允许融合”的预测，并写明哪些前提尚无实测证据。

<details>
<summary>练习答案示例</summary>

FP32 的 32×32×32 GEMM 约有 65,536 次运算，理想输入输出流量 12,288 字节；接一次 ReLU，分步增加中间矩阵一次读与一次写，共 8,192 字节，融合有机会省掉这部分。ReLU 本身的操作应单独计入；给定匹配精度的计算速度 F 与带宽 B，使用 max(运算数/F, 字节数/B) 作简化估算。两种方案的理想复用、融合资源和真实启动成本均待设备验证。

</details>

上游许可见 [LICENSE](https://github.com/NVlabs/SOLAR/blob/8f6e76c1912213e82756217e92d147aedd010f7a/LICENSE)。
