# 理论课程 v1：范围与完成检查

本仓库约定的理论范围是：从 Python 入门到能解释 H20/Hopper 与 SM100 相关 GEMM、Attention 实现的主要数据流、优化取舍和验证方法。**这一范围的章节、图解、练习答案与 CPU 演示已补齐。** 它不等于完成 GPU 实验，也不表示穷尽所有算子、指令变体和上游模板分支。

## 按顺序检查学习结果

每行入口都包含解释或对应练习；“CPU 检查”只证明该教学模型覆盖的数学、索引或协议性质。

| 阶段 | 阅读入口 | 能独立完成什么 | 检查依据 |
| --- | --- | --- | --- |
| Python 到算子 | [先修知识](prerequisites/README.md) | 解释类型、指针、shape、stride、浮点误差 | 各篇自测与答案 |
| 第一个算子 | [向量相加](../lessons/01-cuda-basics/vector-add.md) | 画出线程覆盖、尾部与启动依赖 | check_launch_and_schedule.py |
| 硬件与软件 | [执行](hardware/execution.md)、[存储](hardware/memory.md)、[软件栈](software-stack/README.md) | 分清线程层次、存储位置与编译组件 | 各篇自测 |
| 优化依据 | [资源预算](hardware/resource-budget.md)、[性能估算](methodology/performance-basics.md) | 算空间上界、总任务量和逻辑流量 | check_resource_budget.py；手算答案 |
| 转置与融合 | [经典算子入口](../lessons/02-hardware-and-profiling/README.md) | 分析地址、统计量与中间数组 | check_classic_operators.py |
| 归约、扫描、RoPE | [计算模式](../lessons/02-hardware-and-profiling/operator-patterns.md)及后续章节 | 说明跨线程合并、偏移与成对旋转 | check_patterns.py |
| GEMM 主线 | [GEMM](../lessons/03-gemm/README.md) | 推导分块、精度缩放、split-K 与输出处理 | check_theory.py、check_patterns.py、check_launch_and_schedule.py |
| Attention 前向 | [Attention](../lessons/04-attention/README.md)前四篇 | 手算 m、l、u 与最简梯度 | check_theory.py、check_attention_backward.py |
| 进阶 Attention | [变长与分页](../lessons/04-attention/05-variable-length-and-pages.md)、[合并](../lessons/04-attention/06-split-and-merge.md)、[反向](../lessons/04-attention/07-masked-gqa-backward.md) | 对齐位置、加权合并、累加共享梯度 | check_advanced_attention.py |
| Hopper 机制 | [异步交接](hardware/hopper-pipeline.md) | 区分搬完、算完、可覆盖与可见性 | check_pipeline_protocol.py；阅读题 |
| B 系列机制 | [SM100 矩阵计算](hardware/blackwell-matmul.md) | 解释 TMEM 预算与不同完成条件 | 同上；列预算与答案 |
| 源码与综合 | [逐段阅读](../case_studies/architecture-walkthrough.md)、[案例入口](../case_studies/README.md)、[作业示例](../lessons/05-capstone/worked-example.md) | 将变量和等待对应到理论，写出有边界的结论 | 固定源码链接检查；阅读答案与报告示例 |

## 一条命令检查当前仓库

```bash
python3 tools/check_all.py
```

需要 Python 3.8+ 标准库，不需要 NVIDIA GPU、PyTorch 或浏览器。它运行全部教学检查、固定源码链接结构检查、本地文档与 SVG 图源检查。不要使用 `python -O`，入口会拒绝禁用断言的运行方式。

图形生成另需维护工具；已提交的 SVG 可直接阅读。链接检查不是对所有外部网站在线可用性的保证，CPU 协议模型也不是对 CUDA 内存模型的证明。

## 理论完成后仍待设备环境的工作

- H20 具体型号参数、软件版本和编译目标核验。
- `kernels/` 中完整 CUDA 实现、GPU 边界/竞争检查与库对照。
- Hopper/SM100 实际指令合法性、真实布局与资源报告。
- 代表性输入的时间、端到端开销和原始性能记录。

交互演示是后续教学形式扩展，不阻塞静态理论课程。多 GPU 通信、稀疏/MoE 专用算子、完整服务系统与所有硬件变体不在本轮课程范围；后续扩课单独列目标，避免把它们混作当前章节的未完成项。
