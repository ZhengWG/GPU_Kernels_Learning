"""Small, deterministic CPU exercises for layouts, scans, RoPE and scaling."""

import math
from itertools import accumulate


def blocked_scan(values, block_size):
    local = []
    totals = []
    for start in range(0, len(values), block_size):
        prefix = list(accumulate(values[start:start + block_size]))
        local.append(prefix)
        totals.append(prefix[-1])
    output, offset = [], 0
    for prefix, total in zip(local, totals):
        output.extend(x + offset for x in prefix)
        offset += total
    return output


def rotate_pair(a, b, theta):
    c, s = math.cos(theta), math.sin(theta)
    return a * c - b * s, a * s + b * c


def quantize_integer(values, limit=127):
    scale = max(map(abs, values)) / limit or 1.0
    codes = [max(-limit, min(limit, round(x / scale))) for x in values]
    return codes, scale


def main():
    # The complete 16x4 block must have 64 unique linear thread indices.
    ids = [x + 16 * y for y in range(4) for x in range(16)]
    assert ids == list(range(64))
    assert divmod(3 + 16 * 2, 32) == (1, 3)
    # Each source element of a rectangular transpose must map once.
    m, n = 3, 5
    source = list(range(m * n))
    dest = [None] * len(source)
    for row in range(m):
        for col in range(n):
            dest[col * m + row] = source[row * n + col]
    assert dest == [source[r * n + c] for c in range(n) for r in range(m)]
    for values in [[], [2], [1, 2, 3, 4, 5, 6], [-4, 3, -2, 8, 0]]:
        for size in [1, 2, 3, 8]:
            assert blocked_scan(values, size) == list(accumulate(values))
    print("Layout, rectangular transpose and 16 blocked-scan cases passed.")

    for a, b in [(1., 0.), (3., -4.), (-2., 7.)]:
        for angle in [0., math.pi / 2, .7]:
            x, y = rotate_pair(a, b, angle)
            assert math.isclose(x*x + y*y, a*a + b*b, abs_tol=1e-12)
            restored = rotate_pair(x, y, -angle)
            assert all(math.isclose(x, y, abs_tol=1e-12)
                       for x, y in zip(restored, (a, b)))
    for values in [[0., 0.], [0., 1., 7.], [0., 1., 70.], [-4., .1, 3.]]:
        codes, scale = quantize_integer(values, limit=7)
        assert all(abs(code) <= 7 for code in codes)
        assert all(abs(x - q*scale) <= scale/2 + 1e-12
                   for x, q in zip(values, codes))
    assert quantize_integer([0., 1., 70.], 7)[0] == [0, 0, 7]
    # Compare separately scaled partial products with a dequantized dot product.
    qa, qb, sa, sb = [1, -2, 3], [4, 2, -1], [.5, 2., 1.], [2., .25, 3.]
    reconstructed_a = [q*s for q, s in zip(qa, sa)]
    reconstructed_b = [q*s for q, s in zip(qb, sb)]
    reference = sum(a*b for a, b in zip(reconstructed_a, reconstructed_b))
    grouped = sum(a*b*s*t for a, b, s, t in zip(qa, qb, sa, sb))
    assert math.isclose(grouped, reference)
    print("9 RoPE cases and 4 integer-scaling examples passed.")
    print("CPU mathematics only; integer quantization is not an FP8 simulator.")


if __name__ == "__main__":
    main()
