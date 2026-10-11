"""Reproduce the hypothetical resource worksheet; no device performance model."""
from math import ceil, isclose


def shared_bytes(bm, bn, bk, stages, element_bytes=2):
    if min(bm, bn, bk, stages, element_bytes) <= 0:
        raise ValueError('dimensions, stages and element size must be positive')
    return stages * (bm * bk + bk * bn) * element_bytes


def block_bound(shared, registers_per_thread, threads=256):
    """Worksheet budgets only; excludes hardware allocation granularity."""
    if min(shared, registers_per_thread, threads) <= 0:
        raise ValueError('resource requests must be positive')
    return min(128 * 1024 // shared, 65536 // (registers_per_thread * threads),
               2048 // threads, 32)


def main():
    for label, bm, bn, bk, stages, regs, kib, blocks in [
        ('A', 64, 64, 32, 2, 40, 16, 6),
        ('B', 128, 128, 32, 2, 80, 32, 3),
        ('C', 128, 128, 64, 3, 80, 96, 1),
    ]:
        shared = shared_bytes(bm, bn, bk, stages)
        assert shared == kib * 1024
        assert block_bound(shared, regs) == blocks
        assert bm * bn // 256 <= regs
        intensity = 2 * bm * bn * bk / ((bm * bk + bk * bn) * 2)
        print(f'{label}: input buffers={kib} KiB, block upper bound={blocks}, '
              f'input-only operations/byte={intensity:g}')
    assert block_bound(shared_bytes(128, 128, 32, 4), 80) == 2
    assert block_bound(129 * 1024, 40) == 0
    assert block_bound(16 * 1024, 257) == 0
    assert ceil(256 / 64) ** 2 == 16
    assert ceil(256 / 128) ** 2 == 4
    utilization = 129 ** 2 / ((ceil(129 / 128) * 128) ** 2)
    assert isclose(utilization, 0.2539215087890625)
    print('Worksheet and tail checks passed; hypothetical budgets, not H20 measurements.')


if __name__ == '__main__':
    main()
