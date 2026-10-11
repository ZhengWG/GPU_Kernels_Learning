# 01 CUDA 入门

状态：背景讲解已有初版；vector add 完整实验待实现。

先读 [线程与 GPU 执行](../../docs/hardware/execution.md)，再读 [同步与异步](../../docs/prerequisites/05-synchronization.md)。环境与编译概念见 [软件栈](../../docs/software-stack/README.md)。

学习目标：用线程编号选取元素，说明分配、拷贝、启动、等待与错误检查的顺序。能回答为什么 1000 个元素启动 1024 个线程时必须检查边界。
