#!/usr/bin/env python3
"""Grid-search csAlgo parameters for each KINO spot type (3-12)."""

import json, os, time
import numpy as np
from bisect import bisect_right
from itertools import product

# ─── Load draws ───────────────────────────────────────────────────────────────
draws = []
for yr in range(2024, 2027):
    for mo in range(1, 13):
        fn = f'/home/user/Game/data/raw/kino_raw_{yr}_{mo:02d}.json'
        if os.path.exists(fn):
            with open(fn) as f:
                draws.extend(json.load(f)['draws'])
draws.sort(key=lambda x: x['id'])
N = len(draws)
print(f"Draws loaded: {N}")

# ─── Grid adjacency ───────────────────────────────────────────────────────────
def neighbors(n):
    r,c = (n-1)//10, (n-1)%10
    return [nr*10+nc+1 for dr in [-1,0,1] for dc in [-1,0,1]
            if (dr or dc) and 0<=(r+dr)<=7 and 0<=(c+dc)<=9
            for nr,nc in [(r+dr,c+dc)]]

NBS = {n: neighbors(n) for n in range(1,81)}
NUM_ZONE = np.array([0 if (n-1)//10<3 else 1 if (n-1)//10<6 else 2
                     for n in range(1,81)], dtype=np.int8)

def main_chain(nums):
    ns,vis,best = set(nums),set(),[]
    for s in ns:
        if s in vis: continue
        comp,q = [s],[s]; vis.add(s)
        while q:
            cur=q.pop()
            for nb in NBS[cur]:
                if nb in ns and nb not in vis:
                    vis.add(nb); comp.append(nb); q.append(nb)
        if len(comp)>len(best): best=comp
    return best

def halo_at(chain, depth):
    cs,halo,front = set(chain),set(),set(chain)
    for _ in range(depth):
        nxt = set()
        for n in front:
            for nb in NBS[n]:
                if nb not in cs and nb not in halo: nxt.add(nb)
        halo |= nxt; front = nxt
    return halo

# ─── Appearance index ─────────────────────────────────────────────────────────
app_idx = {n:[] for n in range(1,81)}
for i,d in enumerate(draws):
    for n in d['n']: app_idx[n].append(i)

# ─── Payouts at €0.50 ─────────────────────────────────────────────────────────
RAW_PAY = {
    3:{2:1,3:25},
    4:{2:.5,3:2,4:50},
    5:{3:1,4:10,5:220},
    6:{3:.5,4:2,5:25,6:1000},
    7:{3:.5,4:1.5,5:10,6:50,7:2200},
    8:{4:1,5:5,6:25,7:500,8:2500},
    9:{5:1,6:7,7:50,8:500,9:5000},
    10:{5:1,6:5,7:30,8:200,9:2500,10:25000},
    11:{5:.5,6:3,7:20,8:150,9:1000,10:10000,11:100000},
    12:{6:2.5,7:10,8:75,9:500,10:5000,11:50000,12:500000},
}
pay_arr = {s: np.array([RAW_PAY[s].get(h,0) for h in range(s+1)], dtype=np.float32)
           for s in range(3,13)}

# ─── Test set (last 3000 draws as context, eval against next draw) ────────────
TEST_SIZE = 3000
ctx_start = N - TEST_SIZE - 1
ctx_idx   = np.arange(ctx_start, ctx_start + TEST_SIZE)
eval_idx  = ctx_idx + 1
print(f"Test: {TEST_SIZE} draws | ctx {draws[ctx_start]['id']} → eval {draws[ctx_start+TEST_SIZE]['id']}")

# Hit matrix
hit_matrix = np.zeros((N,80), dtype=np.float32)
for i,d in enumerate(draws):
    for n in d['n']: hit_matrix[i,n-1]=1
eval_hit = hit_matrix[eval_idx]  # (TEST_SIZE, 80)

# ─── Precompute all chains globally (for zone matrices) ───────────────────────
print("Precomputing chains for all draws...", end=' ', flush=True)
t0=time.time()
all_czone = np.zeros(N, dtype=np.int8)
all_csize = np.zeros(N, dtype=np.int8)
for i,d in enumerate(draws):
    ch = main_chain(d['n'])
    all_csize[i] = len(ch)
    if ch:
        zavg = sum((x-1)//10 for x in ch)/len(ch)
        all_czone[i] = 0 if zavg<3 else 1 if zavg<6 else 2
print(f"{time.time()-t0:.1f}s")

# ─── Precompute feature matrices for test draws ───────────────────────────────
print("Precomputing feature matrices...")
t0=time.time()

chain_b = np.zeros((TEST_SIZE,80), dtype=np.float32)
halo1_b = np.zeros((TEST_SIZE,80), dtype=np.float32)
halo2_b = np.zeros((TEST_SIZE,80), dtype=np.float32)
bonus_b = np.zeros((TEST_SIZE,80), dtype=np.float32)

for j,c in enumerate(ctx_idx):
    d = draws[c]
    ch = main_chain(d['n'])
    for n in ch: chain_b[j,n-1]=1
    for n in halo_at(ch,1): halo1_b[j,n-1]=1
    for n in halo_at(ch,2): halo2_b[j,n-1]=1
    b = d.get('b',0)
    if b:
        for nb in NBS[b]: bonus_b[j,nb-1]=1

# Coldness matrices for W in [10, 20, 40]
COLD_WS = [10, 20, 40]
cold_mats = {}
for W in COLD_WS:
    mat = np.empty((TEST_SIZE,80), dtype=np.float32)
    for j,c in enumerate(ctx_idx):
        for n in range(1,81):
            lst = app_idx[n]
            pos = bisect_right(lst,c)-1
            mat[j,n-1] = W if pos<0 else min(W, c-lst[pos])
    cold_mats[W] = mat
    print(f"  cold W={W} ✓ ({time.time()-t0:.1f}s)")

# Zone matrices for (zoneWindow, minChainSize) combos
ZONE_WS  = [5, 10, 20]
MIN_CS   = [3, 6]
zone_mats = {}
for zw,mcs in product(ZONE_WS, MIN_CS):
    mat = np.zeros((TEST_SIZE,80), dtype=np.float32)
    for j,c in enumerate(ctx_idx):
        s = max(0, c-zw+1)
        sizes_sl = all_csize[s:c+1]
        zones_sl = all_czone[s:c+1]
        valid = zones_sl[sizes_sl >= mcs]
        if len(valid)==0: continue
        zh = np.bincount(valid.astype(np.intp), minlength=3)
        cz = int(zh.argmin())
        if zh[cz]==0:
            mat[j, NUM_ZONE==cz] = 1
    zone_mats[(zw,mcs)] = mat
    print(f"  zone ({zw},{mcs}) ✓ ({time.time()-t0:.1f}s)")

print(f"Precomputation total: {time.time()-t0:.1f}s")

# ─── Grid search ──────────────────────────────────────────────────────────────
GRID = {
    'zoneWindow':   [5, 10, 20],
    'coldWindow':   [10, 20, 40],
    'haloBonus':    [0, 8, 16, 24],
    'coldZoneBonus':[0, 8, 16],
    'coldWeight':   [5, 10, 15, 20],   # /10 → 0.5,1.0,1.5,2.0
    'chainPenalty': [0, 8, 16],
    'haloDepth':    [1, 2],
    'bonusWeight':  [0, 5],
    'minChainSize': [3, 6],
}
n_combos = 1
for v in GRID.values(): n_combos *= len(v)
print(f"\nGrid: {n_combos} combos × 10 spots on {TEST_SIZE} draws")

best = {s: {'roi': -1e9, 'pay': 0.0, 'params': None} for s in range(3,13)}
cost_per_test = TEST_SIZE * 0.5

t0 = time.time()
for ci, (zw,cw,hb,czb,cwx,cp,hd,bw,mcs) in enumerate(product(
        GRID['zoneWindow'], GRID['coldWindow'],   GRID['haloBonus'],
        GRID['coldZoneBonus'], GRID['coldWeight'], GRID['chainPenalty'],
        GRID['haloDepth'],    GRID['bonusWeight'], GRID['minChainSize'])):

    cw_float = cwx / 10.0
    halo_mat  = halo2_b if hd==2 else halo1_b

    # Score matrix (TEST_SIZE, 80)
    score = (cold_mats[cw]     * cw_float
           + halo_mat           * hb
           + zone_mats[(zw,mcs)]* czb
           - chain_b            * cp
           + bonus_b            * bw)

    for spots in range(3, 13):
        top_k  = np.argpartition(score, -spots, axis=1)[:, -spots:]  # (TEST_SIZE, spots)
        row_i  = np.repeat(np.arange(TEST_SIZE), spots)
        hits   = eval_hit[row_i, top_k.ravel()].reshape(TEST_SIZE, spots).sum(axis=1)
        hits   = np.clip(hits.astype(np.int32), 0, spots)
        total_pay = pay_arr[spots][hits].sum()
        roi       = (total_pay - cost_per_test) / cost_per_test

        if roi > best[spots]['roi']:
            best[spots].update(roi=roi, pay=float(total_pay), params=dict(
                zoneWindow=zw, coldWindow=cw, haloBonus=hb, coldZoneBonus=czb,
                coldWeight=cwx/10, chainPenalty=cp, haloDepth=hd,
                bonusWeight=bw, minChainSize=mcs))

    if (ci+1) % 1000 == 0:
        elapsed = time.time()-t0
        eta = elapsed/(ci+1)*(n_combos-ci-1)
        print(f"  {ci+1}/{n_combos} ({elapsed:.0f}s elapsed, ~{eta:.0f}s left)")

elapsed = time.time()-t0
print(f"\nGrid search done in {elapsed:.1f}s")

# ─── Print results ────────────────────────────────────────────────────────────
print("\n=== BEST PARAMS PER SPOT TYPE ===")
for s in range(3,13):
    b = best[s]
    print(f"\n{s}-spot | ROI={b['roi']*100:+.1f}% | payout €{b['pay']:.1f} vs cost €{cost_per_test:.0f}")
    print(f"  {b['params']}")

print("\n\n=== JS PRESETS (paste into index.html) ===")
print("const CS_PRESETS = {")
for s in range(3,13):
    p = best[s]['params']
    cw_slider = int(p['coldWeight']*10)
    print(f"  {s}: {{zoneWindow:{p['zoneWindow']},coldWindow:{p['coldWindow']},"
          f"haloBonus:{p['haloBonus']},coldZoneBonus:{p['coldZoneBonus']},"
          f"coldWeight:{p['coldWeight']},coldWeightSlider:{cw_slider},"
          f"chainPenalty:{p['chainPenalty']},haloDepth:{p['haloDepth']},"
          f"bonusWeight:{p['bonusWeight']},minChainSize:{p['minChainSize']}"
          f"}}, // ROI={best[s]['roi']*100:+.1f}%")
print("};")
