# 源码阅读：从课堂公式到固定版本

这些是源码阅读记录，不是性能复现报告。提交号与核对过的文件摘要保存在 [sources.json](sources.json)，所有案例链接固定到对应 commit。

| 案例 | 先完成的理论 | 本轮产出 |
| --- | --- | --- |
| [DeepGEMM](deepgemm/README.md) | 分块、流水线、缩放 | 一条 SM90 FP8 路径的等待与缩放位置 |
| [FlashAttention](flash_attention/README.md) | 逐块 Softmax 与 Attention | m、l、u 到源码变量的对应 |
| [SOLAR](solar/README.md) | 计算量、数据量、性能上界 | 预测公式与假设的对应 |
| [KDA / KDA-Pilot](kda_pilot/README.md) | 正确性与公平计时 | 任务定义和具体案例的测量口径 |

阅读固定版本可以避免主分支更新后目录和行号变化。更新版本时重新核对文件、接口和结论，不能只替换 URL 中的提交号。

## 每次阅读只做四件事

1. 先列输入、输出、尺寸和数据布局。
2. 找到计算循环及线程分工。
3. 标出数据复用、同步和数值缩放。
4. 写明已从源码确认的事实，以及仍需要 GPU 验证的性能假设。

源码读取不等于可移植性验证。SM90 或 SM100 的文件名也不能单独保证某台 H20 或 B 系列设备和当前软件环境可直接运行。

完成四份案例后，阅读 [架构实现逐段对照](architecture-walkthrough.md)：补齐 DeepGEMM 的线程/行布局、Hopper K/V 生命周期和 SM100 的角色/TMEM 分工。
