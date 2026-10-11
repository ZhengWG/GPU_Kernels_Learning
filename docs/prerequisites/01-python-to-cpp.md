# 从 Python 到 C++：先学能看懂 kernel 的部分

**算法可以不变，先把类型、内存和执行位置写清楚。**

## 同一个循环

Python 中，假设 `a`、`b` 是等长列表，`c` 已分配：

```python
for i in range(n):
    c[i] = a[i] + b[i]
```

C++ 中，假设三个指针都指向至少 `n` 个有效的 `float` 元素：

```cpp
for (int i = 0; i < n; ++i) {
    c[i] = a[i] + b[i];
}
```

| 写法 | 白话解释 |
| --- | --- |
| `int i = 0` | 建立整数变量，初值为 0 |
| `float x` | 单精度浮点数；不是 Python 默认浮点精度 |
| `++i` | `i` 增加 1 |
| `{ ... }` | 用大括号围住代码范围 |
| `;` | 结束一条语句 |
| `const float* a` | 通过这个指针读取浮点数，不能通过它修改元素 |

## 指针记录位置，不记录长度

```text
原理示意：每格一个 float

指针 a ──→ [ 2.0 ][ 4.0 ][ 6.0 ][ 8.0 ]
元素索引       0      1      2      3
字节偏移       0      4      8     12

a + 2 指向第三格，*(a + 2) 与 a[2] 表示同一个元素。
```

`float*` 是指针类型，`*p` 是读取指针所指的值，`&x` 是取得变量 `x` 的地址。指针本身不告诉你可以读取多少个元素，因此函数通常还需要 `n`。越界访问不像 Python 列表那样保证给出清晰的索引异常。

`sizeof(float)` 给出一个元素的字节数；分配 `n` 个元素需要 `n * sizeof(float)`。`sizeof(a)` 若作用于指针，只得到指针本身的大小。

还要追问地址属于哪里：普通 CPU 代码不能把 `cudaMalloc` 得到的设备指针当作普通 CPU 数组直接读取。统一内存等其他分配方式后续再学。

## 把循环分给线程

下面只展示 kernel 和启动方式，不是完整程序：

```cpp
__global__ void add(const float* a, const float* b, float* c, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        c[i] = a[i] + b[i];
    }
}

// a、b、c 必须已经指向设备上的有效内存；此例只讨论小尺寸输入。
// n == 0 时跳过启动。
if (n > 0) {
    int threads = 256;
    int blocks = (n + threads - 1) / threads;
    add<<<blocks, threads>>>(a, b, c, n);
}
```

`__global__` 表示可被启动的 GPU 函数；`<<< >>>` 指定线程如何分组。每个线程运行同一份代码，用自己的编号选择一个元素。启动返回不表示计算已经完成。语法与执行规则见 [NVIDIA CUDA C++ 入门](https://docs.nvidia.com/cuda/cuda-programming-guide/02-basics/intro-to-cuda-cpp.html)。

## 暂时只认识这些进阶写法

| 写法 | 何时学 | 用途 |
| --- | --- | --- |
| `__device__` | 抽取 GPU 辅助函数时 | 供 GPU 代码调用 |
| `__shared__` | 线程交换数据时 | 声明块内共享存储 |
| `template<int TILE>` | 分块 GEMM 时 | 编译时选择分块大小 |
| `constexpr` | 需要编译期常量时 | 让编译器提前知道某个值 |

## 自测

1. 四个 `float` 占多少字节？`a[4]` 是否有效？
2. `n = 1000`、每块 256 个线程，需要几个块？为什么还需要 `i < n`？

<details>
<summary>参考答案</summary>

1. 16 字节；四元素数组的有效索引是 0 到 3。
2. 四个块，共 1024 个线程；最后 24 个线程不能访问数组。

</details>

下一篇：[张量与内存布局](02-tensors-and-layout.md)。
