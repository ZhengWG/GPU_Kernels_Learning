# 04 Attention

状态：理论初版已提供；完整 CUDA 实现和 GPU 性能待验证。

1. [逐块 Softmax](01-online-softmax.md)：最大值变化时，怎样修正旧的分母。
2. [分块融合 Attention](02-tiled-attention.md)：同时维护加权和，省去完整分数矩阵的存储。

完成标准：手算分块结果，解释为什么分子分母需要一起缩放，正确处理被屏蔽的位置。无需 GPU。

从仓库根目录运行 `python3 tools/check_theory.py`，只需要 Python 标准库。数学例子包含尾块、大分数和屏蔽输入，不代表 CUDA 正确性验证。
