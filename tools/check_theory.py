"""CPU-only illustrations of tiled GEMM and online attention; no dependencies."""

import math
import random


def gemm_reference(a, b):
    return [[sum(x * y for x, y in zip(row, col)) for col in zip(*b)]
            for row in a]


def gemm_tiled(a, b, bm, bn, bk):
    m, k, n = len(a), len(b), len(b[0])
    out = [[0.0] * n for _ in range(m)]
    for row in range(0, m, bm):
        for col in range(0, n, bn):
            for start in range(0, k, bk):
                for i in range(row, min(row + bm, m)):
                    for j in range(col, min(col + bn, n)):
                        for t in range(start, min(start + bk, k)):
                            out[i][j] += a[i][t] * b[t][j]
    return out


def attention_reference(scores, values, valid):
    """One query row; all-masked output is defined as zero for this demo."""
    ids = [i for i, enabled in enumerate(valid) if enabled]
    if not ids:
        return [0.0] * len(values[0])
    maximum = max(scores[i] for i in ids)
    weights = [math.exp(scores[i] - maximum) for i in ids]
    denom = math.fsum(weights)
    return [math.fsum(w * values[i][d] for w, i in zip(weights, ids)) / denom
            for d in range(len(values[0]))]


def attention_online(scores, values, valid, block_size):
    maximum, denom = None, 0.0
    total = [0.0] * len(values[0])
    for start in range(0, len(scores), block_size):
        ids = [i for i in range(start, min(start + block_size, len(scores)))
               if valid[i]]
        if not ids:
            continue
        block_max = max(scores[i] for i in ids)
        new_max = block_max if maximum is None else max(maximum, block_max)
        scale = 0.0 if maximum is None else math.exp(maximum - new_max)
        weights = [math.exp(scores[i] - new_max) for i in ids]
        denom = scale * denom + sum(weights)
        total = [scale * old + sum(w * values[i][d] for w, i in zip(weights, ids))
                 for d, old in enumerate(total)]
        maximum = new_max
    return [x / denom for x in total] if maximum is not None else total


def check_close(actual, expected):
    if len(actual) != len(expected):
        raise AssertionError("Different output lengths")
    for a, b in zip(actual, expected):
        if not math.isclose(a, b, rel_tol=1e-12, abs_tol=1e-12):
            raise AssertionError(f"{a} != {b}")


def main():
    rng = random.Random(7)
    for m, n, k in [(2, 2, 3), (5, 7, 3), (1, 4, 9)]:
        a = [[rng.uniform(-1, 1) for _ in range(k)] for _ in range(m)]
        b = [[rng.uniform(-1, 1) for _ in range(n)] for _ in range(k)]
        expected = gemm_reference(a, b)
        for tile in [(1, 1, 1), (4, 4, 2), (8, 8, 16)]:
            for actual_row, expected_row in zip(gemm_tiled(a, b, *tile), expected):
                check_close(actual_row, expected_row)
    print("GEMM: 9 shape/tile combinations match, including partial tiles.")

    scores = [0.0, math.log(2), math.log(4)]
    result = attention_online(scores, [[1.0], [3.0], [5.0]], [True] * 3, 2)
    check_close(result, [27 / 7])
    print(f"Hand example: output = {result[0]:.12f} = 27/7.")

    cases = [scores, [1000., 1001., -1000.], [-4.] * 7,
             [rng.uniform(-30, 30) for _ in range(13)]]
    count = 0
    for row in cases:
        values = [[rng.uniform(-2, 2) for _ in range(3)] for _ in row]
        masks = [[True] * len(row), [False] * len(row),
                 [i < 2 for i in range(len(row))],
                 [i == len(row) - 1 for i in range(len(row))]]
        for valid in masks:
            expected = attention_reference(row, values, valid)
            for size in [1, 2, 4, 32]:
                check_close(attention_online(row, values, valid, size), expected)
                count += 1
    print(f"Attention: {count} cases match, including empty and partial masked blocks.")
    print("CPU mathematics only; no CUDA synchronization or performance validated.")


if __name__ == "__main__":
    main()
