# 实验工具

已提供 [check_theory.py](check_theory.py)：使用 Python 标准库验证分块 GEMM 和逐块 Attention 的数学结果。

从仓库根目录执行：

```bash
python3 tools/check_theory.py
```

使用固定随机种子，覆盖不整齐的分块、大分数、重复分数、部分屏蔽和整行屏蔽。教学输入限定为有限数值、非空且尺寸匹配的矩阵；整行屏蔽约定输出零。

该脚本在 CPU 上运行，不模拟线程、缓存和同步，也不测量 GPU 性能。设备检测、GPU 批量验证和计时工具待实际环境可用后提供。
