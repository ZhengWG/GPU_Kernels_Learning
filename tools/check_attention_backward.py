"""Check single-head attention gradients against CPU finite differences."""

import math
import random


def transpose(a):
    return [list(row) for row in zip(*a)]


def matmul(a, b):
    return [[sum(x*y for x, y in zip(row, col)) for col in zip(*b)] for row in a]


def scaled(a, s):
    return [[x*s for x in row] for row in a]


def forward(q, k, v):
    scores = scaled(matmul(q, transpose(k)), 1 / math.sqrt(len(q[0])))
    p = []
    for row in scores:
        weights = [math.exp(x - max(row)) for x in row]
        p.append([x/sum(weights) for x in weights])
    return matmul(p, v), p


def backward(q, k, v, do):
    _, p = forward(q, k, v)
    dv = matmul(transpose(p), do)
    dp = matmul(do, transpose(v))
    ds = []
    for probabilities, grads in zip(p, dp):
        dot = sum(a*b for a, b in zip(probabilities, grads))
        ds.append([a*(b-dot) for a, b in zip(probabilities, grads)])
    scale = 1 / math.sqrt(len(q[0]))
    return scaled(matmul(ds, k), scale), scaled(matmul(transpose(ds), q), scale), dv


def main():
    rng = random.Random(11)
    make = lambda rows, cols: [[rng.uniform(-.7, .7) for _ in range(cols)]
                              for _ in range(rows)]
    checked = 0
    for lq, lk, d, dv_size in [(2, 3, 2, 2), (1, 2, 3, 1), (3, 1, 2, 2)]:
        q, k, v, do = make(lq, d), make(lk, d), make(lk, dv_size), make(lq, dv_size)

        def objective():
            out, _ = forward(q, k, v)
            return sum(x*g for row, grad in zip(out, do) for x, g in zip(row, grad))

        analytic = backward(q, k, v, do)
        for tensor, gradient in zip((q, k, v), analytic):
            for i, row in enumerate(tensor):
                for j, old in enumerate(row):
                    h = 1e-5
                    try:
                        row[j] = old + h
                        plus = objective()
                        row[j] = old - h
                        minus = objective()
                    finally:
                        row[j] = old
                    numerical = (plus - minus) / (2*h)
                    if not math.isclose(numerical, gradient[i][j], rel_tol=1e-6, abs_tol=1e-8):
                        raise AssertionError((numerical, gradient[i][j]))
                    checked += 1
    print(f"Attention backward: {checked} input derivatives match finite differences.")
    print("CPU, finite inputs, single head, no mask/dropout; not CUDA validation.")


if __name__ == "__main__":
    main()
