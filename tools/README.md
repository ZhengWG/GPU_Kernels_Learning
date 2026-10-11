# 实验工具

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
