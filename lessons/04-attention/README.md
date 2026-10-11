# 04 Attention

状态：理论初版已提供；完整 CUDA 实现和 GPU 性能待验证。

1. [逐块 Softmax](01-online-softmax.md)：最大值变化时修正旧分母。
2. [分块融合 Attention](02-tiled-attention.md)：维护加权和，省去完整分数矩阵的存储。
3. [Prefill、Decode 与 GQA](03-prefill-decode-gqa.md)：理解查询数量、缓存与头共享。
4. [反向推导](04-backward.md)：把变化传回 Q、K、V，用数值方法检查导数。
5. [FlashAttention 固定版本阅读](../../case_studies/flash_attention/README.md)：找到 m、l、u 的源码对应。

完成标准：手算分块结果，解释分子分母的缩放，正确处理屏蔽位置；理解最简反向计算。无需 GPU。

运行 `python3 tools/check_theory.py` 和 `python3 tools/check_attention_backward.py`，只需要 Python 标准库。检查通过不代表 CUDA 正确性验证。
