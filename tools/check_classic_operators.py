"""CPU teaching demos: transpose indexing, row normalization and fusion.

Run from the repo root with Python 3.8+. No third-party packages or GPU needed.
The transpose models load/store phases, not concurrent execution. Python float
is double precision; Decimal provides a separate normalization reference.
"""
import math
import random
import struct
from collections import Counter
from decimal import Decimal, localcontext


def transpose_tiled(src, m, n, src_stride, dst_stride, tile=32, padding=1):
    """Out-of-place transpose with distinct loading and storing owners."""
    if min(m, n, tile) <= 0 or src_stride < n or dst_stride < m or padding < 0:
        raise ValueError('positive shape/tile and sufficient row strides required')
    if len(src) != m * src_stride:
        raise ValueError('source storage must include complete padded rows')
    dst = [None] * (n * dst_stride)
    writes = [0] * len(dst)
    for r0 in range(0, m, tile):
        for c0 in range(0, n, tile):
            shared = [None] * (tile * (tile + padding))
            # Phase 1: tx varies across input columns.
            for ty in range(tile):
                for tx in range(tile):
                    r, c = r0 + ty, c0 + tx
                    if r < m and c < n:
                        shared[ty * (tile + padding) + tx] = src[r * src_stride + c]
            # Phase boundary models a completed block-wide wait.
            # Phase 2: tx varies across OUTPUT columns.
            for ty in range(tile):
                for tx in range(tile):
                    out_r, out_c = c0 + ty, r0 + tx
                    if out_r < n and out_c < m:
                        value = shared[tx * (tile + padding) + ty]
                        assert value is not None, 'read before load'
                        address = out_r * dst_stride + out_c
                        dst[address] = value
                        writes[address] += 1
    for r in range(n):
        assert writes[r * dst_stride:r * dst_stride + m] == [1] * m
        assert dst[r * dst_stride + m:(r + 1) * dst_stride] == [None] * (dst_stride - m)
    return dst


def centered_stats(row):
    mean = math.fsum(row) / len(row)
    variance = math.fsum((x - mean) ** 2 for x in row) / len(row)
    return mean, variance


def merge_stats(a, b):
    """Merge (count, mean, sum of centered squares), including empty groups."""
    na, ma, m2a = a
    nb, mb, m2b = b
    if not na:
        return b
    if not nb:
        return a
    delta, n = mb - ma, na + nb
    return n, ma + delta * nb / n, m2a + m2b + delta * delta * na * nb / n


def grouped_stats(row, group):
    state = (0, 0., 0.)
    for start in range(0, len(row), group):
        part = (0, 0., 0.)
        for x in row[start:start + group]:
            part = merge_stats(part, (1, x, 0.))
        state = merge_stats(state, part)
    n, mean, m2 = state
    assert n == len(row)
    return mean, m2 / n


def normalize(row, gamma, beta, eps=1e-5, rms=False, group=None):
    if not row or eps <= 0 or len(gamma) != len(row) or len(beta) != len(row):
        raise ValueError('nonempty row, matching parameters and positive epsilon required')
    if rms:
        mean = 0.
        # Group sums must divide by valid count, never by padded width.
        size = group or len(row)
        total = math.fsum(math.fsum(x*x for x in row[i:i+size])
                          for i in range(0, len(row), size))
        variance = total / len(row)
    else:
        mean, variance = centered_stats(row) if group is None else grouped_stats(row, group)
    denominator = math.sqrt(variance + eps)
    return [(x-mean)/denominator*g + (0. if rms else b)
            for x, g, b in zip(row, gamma, beta)]


def decimal_reference(row, gamma, beta, eps=1e-5, rms=False):
    with localcontext() as ctx:
        ctx.prec = 60
        x = [Decimal.from_float(v) for v in row]
        d = Decimal(len(x))
        mean = Decimal(0) if rms else sum(x) / d
        variance = sum((v-mean)**2 for v in x) / d
        denom = (variance + Decimal.from_float(eps)).sqrt()
        return [float((v-mean)/denom*Decimal.from_float(g) +
                      (Decimal(0) if rms else Decimal.from_float(b)))
                for v, g, b in zip(x, gamma, beta)]


def f32(value):
    return struct.unpack('f', struct.pack('f', value))[0]


def residual_rms(x, residual, gamma, mode, eps=1e-5, cast=float):
    """Keep the same intermediate rounding in all three implementations."""
    if mode == 'separate':
        intermediate = [cast(a+b) for a, b in zip(x, residual)]
        return normalize(intermediate, gamma, [0.] * len(x), eps, rms=True)
    if mode == 'keep':
        held = [cast(a+b) for a, b in zip(x, residual)]
        square_sum = math.fsum(v*v for v in held)
        values = held
    elif mode == 'recompute':
        square_sum = math.fsum(cast(a+b)**2 for a, b in zip(x, residual))
        values = (cast(a+b) for a, b in zip(x, residual))
    else:
        raise ValueError('unknown mode')
    denom = math.sqrt(square_sum / len(x) + eps)
    return [v/denom*g for v, g in zip(values, gamma)]


def close(actual, expected, tolerance=1e-10):
    assert len(actual) == len(expected)
    assert all(math.isclose(a, b, rel_tol=tolerance, abs_tol=tolerance)
               for a, b in zip(actual, expected)), (actual, expected)


def main():
    transpose_cases = 0
    for m, n in [(1, 1), (1, 33), (33, 1), (3, 5), (31, 33), (32, 32), (65, 37)]:
        for extra in [0, 3]:
            ss, ds = n + extra, m + extra
            src = [None] * (m * ss)
            for r in range(m):
                src[r*ss:r*ss+n] = [r*n+c for c in range(n)]
            for tile in [4, 32]:
                for padding in [0, 1]:
                    result = transpose_tiled(src, m, n, ss, ds, tile, padding)
                    expected = [src[r*ss+c] for c in range(n) for r in range(m)]
                    actual = [result[c*ds+r] for c in range(n) for r in range(m)]
                    assert actual == expected
                    transpose_cases += 1
    for c in range(32):
        crowded = Counter((t*32+c) % 32 for t in range(32))
        spread = Counter((t*33+c) % 32 for t in range(32))
        assert max(crowded.values()) == 32 and max(spread.values()) == 1
    print(f'Transpose: {transpose_cases} shape/stride/tile/padding cases; bank mapping passed.')

    rng = random.Random(20261011)
    rows = [[0.], [7.], [0.] * 33, [4.] * 7, [1., 3.],
            [1e8 + i for i in range(9)]]
    rows += [[rng.uniform(-5, 5) for _ in range(d)] for d in [3, 31, 32, 33, 257]]
    checks = 0
    for row in rows:
        gamma = [0.5 + (i % 5) * .25 for i in range(len(row))]
        beta = [(i % 3) * .2 - .1 for i in range(len(row))]
        for rms in [False, True]:
            reference = decimal_reference(row, gamma, beta, rms=rms)
            for group in [None, 1, 4, 32]:
                close(normalize(row, gamma, beta, rms=rms, group=group), reference, 1e-7)
                checks += 1
        if len(set(row)) == 1:
            close(normalize(row, gamma, beta), beta)
    # Deliberate bad variance formulation, compared with an exact expected value.
    offset_row = [1e8 + i for i in range(4)]
    unstable = math.fsum(x*x for x in offset_row)/4 - (math.fsum(offset_row)/4)**2
    stable = centered_stats(offset_row)[1]
    assert stable == 1.25 and abs(unstable - stable) > .1
    assert merge_stats((0, 0., 0.), (2, 2., 2.)) == (2, 2., 2.)
    assert merge_stats((2, 2., 2.), (0, 0., 0.)) == (2, 2., 2.)
    assert merge_stats((2, 2., 2.), (1, 5., 0.)) == (3, 3., 8.)
    print(f'Normalization: {checks} comparisons with Decimal passed.')
    print(f'Cancellation example: subtracting large squares gives {unstable:g}; centered variance={stable:g}.')

    fusion_cases = 0
    for d in [1, 3, 33, 257]:
        x = [rng.uniform(-5, 5) for _ in range(d)]
        residual = [rng.uniform(-5, 5) for _ in range(d)]
        gamma = [rng.uniform(.5, 1.5) for _ in range(d)]
        for cast in [float, f32]:
            ref = decimal_reference([cast(a+b) for a, b in zip(x, residual)],
                                    gamma, [0.] * d, rms=True)
            for mode in ['separate', 'keep', 'recompute']:
                close(residual_rms(x, residual, gamma, mode, cast=cast), ref)
                fusion_cases += 1
        # Multiple rows exercise the per-column bias indexing.
        rows_x = [x, list(reversed(x))]
        tmp = [[f32(a+b) for a, b in zip(row, residual)] for row in rows_x]
        separate = [[max(v, 0.) for v in row] for row in tmp]
        fused = [[max(f32(a+b), 0.) for a, b in zip(row, residual)] for row in rows_x]
        assert separate == fused
    assert f32(16777216. + 1.) == 16777216.
    assert 16777216. + 1. == 16777217.
    md = 3 * 33
    assert 6*md - 4*md == 2*md  # bias + ReLU intermediate write/read
    assert 7*md - 4*md == 3*md  # residual RMS, specified two-pass baseline
    print(f'Fusion: {fusion_cases} residual RMS comparisons, bias/ReLU and rounding checks passed.')
    print('CPU math/index checks only; no CUDA synchronization, FP8 simulation or GPU timing.')


if __name__ == '__main__':
    main()
