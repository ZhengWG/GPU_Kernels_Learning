"""CPU reference checks: split attention, packed/pages, masked GQA gradients.

Finite inputs only. Empty attention rows return zeros. Dropout factors are
fixed during differentiation; no GPU RNG, cache allocator or concurrency model.
"""
import math
import random
from itertools import permutations


def close(a, b, tol=1e-9):
    assert len(a) == len(b)
    assert all(math.isclose(x,y,rel_tol=tol,abs_tol=tol) for x,y in zip(a,b)), (a,b)


def probabilities(scores, allowed):
    active = [s for s, keep in zip(scores,allowed) if keep]
    if not active:
        return [0.]*len(scores)
    peak = max(active)
    weights = [math.exp(s-peak) if keep else 0. for s,keep in zip(scores,allowed)]
    total = math.fsum(weights)
    return [w/total for w in weights]


def direct(scores, values, allowed, dv):
    p = probabilities(scores,allowed)
    return [math.fsum(w*v[c] for w,v in zip(p,values)) for c in range(dv)]


def summarize(scores, values, allowed, dv):
    active = [(s,v) for s,v,keep in zip(scores,values,allowed) if keep]
    if not active:
        return None
    m = max(s for s,_ in active)
    weights = [math.exp(s-m) for s,_ in active]
    l = math.fsum(weights)
    u = [math.fsum(w*v[c] for w,(_,v) in zip(weights,active)) for c in range(dv)]
    return m,l,u


def merge(a,b):
    if a is None:
        return b
    if b is None:
        return a
    ma,la,ua = a
    mb,lb,ub = b
    m = max(ma,mb)
    wa,wb = math.exp(ma-m),math.exp(mb-m)
    return m,wa*la+wb*lb,[wa*x+wb*y for x,y in zip(ua,ub)]


def finish(state,dv):
    return [0.]*dv if state is None else [u/state[1] for u in state[2]]


def merge_normalized(states,dv):
    valid = [s for s in states if s is not None]
    if not valid:
        return [0.]*dv
    logs = [m+math.log(l) for m,l,_ in valid]
    weights = [math.exp(x-max(logs)) for x in logs]
    return [math.fsum(w*u[c]/l for w,(_,l,u) in zip(weights,valid))/math.fsum(weights)
            for c in range(dv)]


def forward(q,k,v,mask,factors):
    """Tensors are [head][position][feature]; mask is [query][key]."""
    if len(q) % len(k):
        raise ValueError('query heads must divide into whole KV groups')
    group = len(q)//len(k)
    scale = 1/math.sqrt(len(q[0][0]))
    dv = len(v[0][0])
    out,probs = [],[]
    for h,queries in enumerate(q):
        kv = h//group
        head_out,head_p = [],[]
        for i,query in enumerate(queries):
            scores = [sum(a*b for a,b in zip(query,key))*scale for key in k[kv]]
            p = probabilities(scores,mask[i])
            head_p.append(p)
            head_out.append([sum(p[j]*factors[h][i][j]*v[kv][j][c]
                                 for j in range(len(p))) for c in range(dv)])
        out.append(head_out)
        probs.append(head_p)
    return out,probs


def zeros_like(tensor):
    return [[[0.]*len(row) for row in head] for head in tensor]


def backward(q,k,v,mask,factors,do):
    _,probs = forward(q,k,v,mask,factors)
    dq,dk,dv = (zeros_like(t) for t in (q,k,v))
    group,scale = len(q)//len(k),1/math.sqrt(len(q[0][0]))
    for h,queries in enumerate(q):
        kv = h//group
        for i,query in enumerate(queries):
            p = probs[h][i]
            dp = [sum(a*b for a,b in zip(do[h][i],value))*factors[h][i][j]
                  for j,value in enumerate(v[kv])]
            row_dot = sum(a*b for a,b in zip(p,dp))
            ds = [a*(b-row_dot) for a,b in zip(p,dp)]
            for j in range(len(k[kv])):
                if not mask[i][j]:
                    assert ds[j] == 0.
                for c in range(len(query)):
                    dq[h][i][c] += ds[j]*k[kv][j][c]*scale
                    dk[kv][j][c] += ds[j]*query[c]*scale
                for c in range(len(v[kv][j])):
                    dv[kv][j][c] += p[j]*factors[h][i][j]*do[h][i][c]
    return dq,dk,dv


def gather_pages(cache,table,length,page_size):
    if length < 0 or page_size <= 0 or len(table)*page_size < length:
        raise ValueError('invalid length/page table')
    return [cache[table[j//page_size]][j%page_size] for j in range(length)]


def pack(sequences):
    flat,offsets = [],[0]
    for seq in sequences:
        flat.extend(seq)
        offsets.append(len(flat))
    return flat,offsets


def main():
    rng = random.Random(37)
    split_checks = 0
    for n in [0,1,2,5,17]:
        scores = [rng.uniform(-4,4) for _ in range(n)]
        values = [[rng.uniform(-2,2),rng.uniform(-2,2)] for _ in scores]
        for shift in [0.,1000.,-1000.]:
            shifted = [x+shift for x in scores]
            for mask in [[True]*n,[False]*n,[i%2==0 for i in range(n)]]:
                ref = direct(shifted,values,mask,2)
                for size in [1,3,8]:
                    states = [None] + [summarize(shifted[s:s+size],values[s:s+size],mask[s:s+size],2)
                                       for s in range(0,n,size)] + [None]
                    for ordered in [states,list(reversed(states))]:
                        state = None
                        for s in ordered:
                            state = merge(state,s)
                        close(finish(state,2),ref)
                        close(merge_normalized(ordered,2),ref)
                        split_checks += 1
    # Every order of a three-way merge agrees; direct mean is deliberately wrong.
    hand = [summarize([s],[[v]],[True],1) for s,v in [(0.,2.),(math.log(3),10.)]]
    for order in permutations(hand+[None]):
        state = None
        for part in order:
            state = merge(state,part)
        close(finish(state,1),[8.])
    assert (2+10)/2 != 8
    print(f'Split attention: {split_checks} order/shape/mask checks and hand example passed.')

    page_checks = 0
    for page_size in [1,2,4]:
        for n in [0,1,5,9]:
            count = (n+page_size-1)//page_size
            table = list(reversed(range(count)))
            cache = [[[999.,-999.] for _ in range(page_size)] for _ in range(count)]
            contiguous = [[float(j),float(j%3-1)] for j in range(n)]
            for j,value in enumerate(contiguous):
                cache[table[j//page_size]][j%page_size] = value
            actual = gather_pages(cache,table,n,page_size)
            assert actual == contiguous
            q = [.2,-.1]
            scores = lambda vv:[sum(a*b for a,b in zip(q,v)) for v in vv]
            close(direct(scores(actual),actual,[True]*n,2),
                  direct(scores(contiguous),contiguous,[True]*n,2))
            page_checks += 1
    cache = [[[90.,90.] for _ in range(2)] for _ in range(4)]
    for j in range(5):
        cache[[2,0,3][j//2]][j%2] = [float(j),-float(j)]
    assert gather_pages(cache,[2,0,3],5,2) == [[float(j),-float(j)] for j in range(5)]
    seqs = [[[1.],[2.]],[],[[100.],[101.],[102.]]]
    packed,offsets = pack(seqs)
    assert offsets == [0,2,2,5]
    for i,seq in enumerate(seqs):
        extracted = packed[offsets[i]:offsets[i+1]]
        assert extracted == seq
        close(direct([v[0]*.01 for v in extracted],extracted,[True]*len(seq),1),
              direct([v[0]*.01 for v in seq],seq,[True]*len(seq),1))
    print(f'Paging: {page_checks+1} layouts; packed empty sequence and batch isolation passed.')

    derivatives = 0
    # Rectangular right-aligned causal, window, all-masked and fixed-dropout cases.
    cases = [(4,2,2,3,False,'causal'),(2,1,3,2,False,'causal'),
             (4,2,2,3,True,'window'),(2,1,2,2,True,'empty')]
    for hq,hkv,nq,nk,drop,kind in cases:
        make = lambda heads,rows,cols:[[[rng.uniform(-.5,.5) for _ in range(cols)]
                                        for _ in range(rows)] for _ in range(heads)]
        q,k,v,do = make(hq,nq,2),make(hkv,nk,2),make(hkv,nk,2),make(hq,nq,2)
        mask = [[(j<=nk-nq+i) and (kind!='window' or j>=nk-nq+i-1)
                 and kind!='empty' for j in range(nk)] for i in range(nq)]
        factors = [[[((0. if (h+i+j)%3==0 else 1.25) if drop else 1.)
                     for j in range(nk)] for i in range(nq)] for h in range(hq)]
        def loss():
            out,_ = forward(q,k,v,mask,factors)
            return sum(x*g for head,grad in zip(out,do) for row,gr in zip(head,grad)
                       for x,g in zip(row,gr))
        analytic = backward(q,k,v,mask,factors,do)
        for tensor,gradient in zip((q,k,v),analytic):
            for h,head in enumerate(tensor):
                for i,row in enumerate(head):
                    for c,old in enumerate(row):
                        step = 1e-5
                        try:
                            row[c] = old+step
                            plus = loss()
                            row[c] = old-step
                            minus = loss()
                        finally:
                            row[c] = old
                        close([(plus-minus)/(2*step)],[gradient[h][i][c]],2e-7)
                        derivatives += 1
        if kind == 'empty':
            assert all(x==0 for tensor in analytic for head in tensor for row in head for x in row)
    print(f'Masked GQA/dropout: {derivatives} derivatives match finite differences.')
    print('CPU finite-input model; no CUDA, dropout RNG or cache-update concurrency validated.')


if __name__ == '__main__':
    main()
