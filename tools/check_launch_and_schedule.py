"""CPU checks for launch indexing, task coverage and split-K output handling."""
from collections import Counter
from math import ceil, isclose
import random


def vector_add(x, z, threads, blocks=None):
    if len(x) != len(z) or threads <= 0 or (blocks is not None and blocks <= 0):
        raise ValueError('matching arrays and positive launch dimensions required')
    if not x:
        return []
    blocks = blocks or ceil(len(x) / threads)
    out, hits = [None]*len(x), [0]*len(x)
    for block in range(blocks):
        for thread in range(threads):
            for i in range(block*threads+thread, len(x), blocks*threads):
                out[i] = x[i] + z[i]
                hits[i] += 1
    assert hits == [1]*len(x)
    return out


def split_gemm(a, b, split_size, bias):
    m, k, n = len(a), len(b), len(b[0])
    partials = [[[sum(a[i][t]*b[t][j] for t in range(start, min(start+split_size, k)))
                  for j in range(n)] for i in range(m)] for start in range(0, k, split_size)]
    return [[max(sum(p[i][j] for p in partials)+bias[j], 0.) for j in range(n)] for i in range(m)]


def main():
    checks = 0
    for n in [0, 1, 9, 31, 32, 33, 1000]:
        x, z = list(range(n)), [-i*.5 for i in range(n)]
        for t in [4, 32, 256]:
            for blocks in [None, 1, 2, 5]:
                assert vector_add(x, z, t, blocks) == [a+b for a, b in zip(x,z)]
                checks += 1
    for count in [0, 1, 8, 35]:
        for workers in [1, 3, 16]:
            assigned = [task for w in range(workers) for task in range(w, count, workers)]
            assert Counter(assigned) == Counter(range(count))
    rng = random.Random(29)
    for m, n, k in [(1,1,1), (2,3,7), (5,2,9)]:
        a = [[rng.uniform(-1,1) for _ in range(k)] for _ in range(m)]
        b = [[rng.uniform(-1,1) for _ in range(n)] for _ in range(k)]
        bias = [rng.uniform(-1,1) for _ in range(n)]
        ref = [[max(sum(a[i][t]*b[t][j] for t in range(k))+bias[j],0.) for j in range(n)] for i in range(m)]
        for size in [1, 2, 4, 16]:
            out = split_gemm(a,b,size,bias)
            assert all(isclose(x,y,rel_tol=1e-12,abs_tol=1e-12)
                       for row, rr in zip(out,ref) for x,y in zip(row,rr))
    assert max(-2+3,0) != max(-2,0)+max(3,0)
    assert 4+5+2 != (4+2)+(5+2)
    print(f'Launch: {checks} cases; scheduling: 12 task sets; split-K: 12 comparisons passed.')
    print('CPU index/math only; no GPU scheduling or timings.')


if __name__ == '__main__':
    main()
