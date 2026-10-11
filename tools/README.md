# 实验工具

统一入口：`python3 tools/check_all.py`，使用当前 Python 运行全部无 GPU 检查。各工具的检查边界如下。

已提供 [check_theory.py](check_theory.py)：使用 Python 标准库验证分块 GEMM 和逐块 Attention 的数学结果。

从仓库根目录执行：

```bash
python3 tools/check_theory.py
```

使用固定随机种子，覆盖不整齐的分块、大分数、重复分数、部分屏蔽和整行屏蔽。教学输入限定为有限数值、非空且尺寸匹配的矩阵；整行屏蔽约定输出零。

该脚本在 CPU 上运行，不模拟线程、缓存和同步，也不测量 GPU 性能。设备检测、GPU 批量验证和计时工具待实际环境可用后提供。

[check_patterns.py](check_patterns.py) 检查线程编号、转置、分块前缀和、RoPE 和整数缩放：`python3 tools/check_patterns.py`。它不是 FP8 模拟器。

[check_attention_backward.py](check_attention_backward.py) 用有限差分检查最简 Attention 的 Q/K/V 导数。

[check_docs.py](check_docs.py) 检查本地文档链接、代码块和 SVG 图源是否同步，运行 `python3 tools/check_docs.py`。

[render_diagrams.py](render_diagrams.py) 使用固定版本 Mermaid 10.9.1 浏览器文件生成静态 SVG，配置见 [mermaid-config.json](mermaid-config.json)。仅绘图维护需要外部工具，CPU 检查没有第三方依赖。

[check_resource_budget.py](check_resource_budget.py) 复算三个分块候选、驻留块上界和尾块比例。预算是教学假设，没有检测 H20，也没有实现完整硬件资源计算器。

[check_classic_operators.py](check_classic_operators.py) 演示分阶段转置、两遍/分组归一化与融合；用 Decimal 独立参考检查数值，覆盖尾块、行填充、常数行和中间 FP32 舍入。运行：

```bash
python3 tools/check_resource_budget.py
python3 tools/check_classic_operators.py
```

检查全部采用固定输入或随机种子。不要用 `python -O` 运行这些教学检查，因为它会禁用断言。

[check_launch_and_schedule.py](check_launch_and_schedule.py)：向量相加的覆盖、尾部、任务分配及 split-K 输出处理。

[check_advanced_attention.py](check_advanced_attention.py)：跨块合并、空段、变长边界、分页寻址、掩码/GQA/固定 dropout 梯度。

[check_pipeline_protocol.py](check_pipeline_protocol.py)：抽象缓冲区交接与错误顺序、TMEM 列预算；不模拟 PTX 或 CUDA 内存可见性。

[check_source_links.py](check_source_links.py)：离线检查案例链接的提交号与路径。若本地有对应上游文件，运行 `python3 tools/check_source_links.py --checkout-root /path/to/checkouts` 进一步核对文件摘要和行号范围；子目录名须为 DeepGEMM、flash-attention、SOLAR、kda、KDA-Pilot。
