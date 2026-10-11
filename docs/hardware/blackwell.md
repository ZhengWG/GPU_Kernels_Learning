# B 系列扩展

状态：待选定具体型号，暂无验证记录。

按具体 GPU 分别介绍硬件能力、与 H20 实现的差异、所需软件版本和上游案例。只有在对应设备运行后才能标记已验证。

## 型号需要分开记录

| 型号 | NVIDIA 列出的计算能力 | 本仓库实测 |
| --- | --- | --- |
| B200 | 10.0 | 无 |
| B300 | 10.3 | 无 |

来源：[NVIDIA GPU 计算能力表](https://developer.nvidia.com/cuda/gpus)，访问日期 2026-10-11。计算能力是功能版本，不表示相对速度。

扩展时先确认编译兼容，再检查共享内存、寄存器与块分组限制，最后分析矩阵计算和搬运方式。B200 的参数不能自动用于所有 Blackwell 设备。[Blackwell Tuning Guide](https://docs.nvidia.com/cuda/blackwell-tuning-guide/index.html)

课程选择 [SM100 矩阵计算](blackwell-matmul.md) 作为具体理论路径，解释 TMEM、单/双块参与和完成条件；不据此宣称所有 B 系列已验证。
