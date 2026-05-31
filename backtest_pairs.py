#!/usr/bin/env python3
"""
Approach 1: Pair co-occurrence backtest
Two prediction modes tested against NoTimeBias baseline:
  A) PairDelay   - delay groups (same as before) but sort row selection by pair affinity
  B) PairCond    - given current draw, pick 8 with highest co-occurrence with those numbers
"""
import json, time, numpy as np
from pathlib import Path

DATA_DIR = Path('/home/user/Game/data/raw')
W = 60
DIST_CFG = [3,0,2,1,0,2,0,0,0,0]  # best 8-spot from delay backtest

# ── 1. Load ────────────────────────────────────────────────────────────────
print("Loading draws...")
t0 = time.time()
all_draws = []
for f in sorted(DATA_DIR.glob('kino_raw_*.json')):
    with open(f) as fp:
        data = json.load(fp)
    for d in data.get('draws', []):
        all_draws.append((d['id'], frozenset(d['n'])))

all_draws.sort(key=lambda x: x[0])
draw_nums = [d[1] for d in all_draws]
N = len(all_draws)
print(f"  {N} draws in {time.time()-t0:.1f}s")

# ── 2. Global pair co-occurrence matrix ────────────────────────────────────
print("Computing pair co-occurrence matrix...")
t0 = time.time()
pair_mat = np.zeros((81, 81), dtype=np.float32)  # 1-indexed, [0] unused

for draw in draw_nums:
    nums = list(draw)
    for ii in range(len(nums)):
        for jj in range(ii+1, len(nums)):
            pair_mat[nums[ii]][nums[jj]] += 1
            pair_mat[nums[jj]][nums[ii]] += 1

pair_freq = pair_mat / N  # normalize to frequency
print(f"  Done in {time.time()-t0:.1f}s")

# Baseline and top pairs
baseline_rate = 20*19 / (80*79)
p = baseline_rate
sigma = (N * p * (1-p)) ** 0.5
expected = N * p

print(f"\n  Baseline expected per pair: {expected:.0f}  σ={sigma:.1f}")

top_pairs = []
for i in range(1, 81):
    for j in range(i+1, 81):
        cnt = pair_mat[i][j]
        z = (cnt - expected) / sigma
        top_pairs.append((z, int(cnt), i, j))
top_pairs.sort(reverse=True)

print(f"\nTop 20 pairs (z-score above baseline):")
print(f"  {'Pair':>10}  {'Count':>7}  {'Expected':>9}  {'z':>7}  {'Grid adj?':>10}")
for z, cnt, i, j in top_pairs[:20]:
    ri, ci_ = (i-1)//10, (i-1)%10
    rj, cj_ = (j-1)//10, (j-1)%10
    adj = abs(ri-rj) <= 1 and abs(ci_-cj_) <= 1
    print(f"  ({i:2d},{j:2d})      {cnt:>7}  {expected:>9.0f}  {z:>+7.2f}  {'YES' if adj else 'no':>10}")

print(f"\nBottom 5 pairs (least co-occurring):")
for z, cnt, i, j in top_pairs[-5:]:
    print(f"  ({i:2d},{j:2d})  count={cnt}  z={z:+.2f}")

# ── 3. Precompute delay table ──────────────────────────────────────────────
print("\nPrecomputing delay table...")
t0 = time.time()
last_seen = {n: -W for n in range(1, 81)}
by_row_first = []

for i, draw in enumerate(draw_nums):
    entry = [{} for _ in range(10)]
    for n in range(1, 81):
        d = 0 if n in draw else min(W, i - last_seen[n])
        g = min(9, d)
        r = (n-1) // 10
        if r not in entry[g]:
            entry[g][r] = n
    by_row_first.append(entry)
    for n in draw:
        last_seen[n] = i

print(f"  Done in {time.time()-t0:.1f}s")

# ── 4. Simulation functions ────────────────────────────────────────────────
def sim_nodelay_base(i):
    """Baseline: delay groups, unused rows first (sorted)."""
    entry = by_row_first[i]
    picks, used_rows = [], set()
    for gi in range(10):
        count = DIST_CFG[gi]
        if count <= 0: continue
        grp = entry[gi]
        rows = (sorted(r for r in grp if r not in used_rows) +
                sorted(r for r in grp if r in used_rows))
        picked = 0
        for r in rows:
            if picked >= count: break
            picks.append(grp[r])
            used_rows.add(r)
            picked += 1
    return frozenset(picks)


def sim_pair_delay(i):
    """PairDelay: same delay groups, but sort rows by pair affinity within candidates."""
    entry = by_row_first[i]

    # All candidate numbers (one per row per group)
    all_cands = []
    for gi in range(10):
        for r, n in entry[gi].items():
            all_cands.append(n)

    # Score each candidate: sum of pair_freq with all other candidates
    cand_arr = np.array(all_cands)
    # pair_freq[n, cand_arr] = row n of pair_freq at candidate indices
    cand_scores = {}
    for gi in range(10):
        for r, n in entry[gi].items():
            cand_scores[(gi, r)] = float(pair_freq[n, cand_arr].sum() - pair_freq[n, n])

    picks, used_rows = [], set()
    for gi in range(10):
        count = DIST_CFG[gi]
        if count <= 0: continue
        grp = entry[gi]
        unused = sorted((r for r in grp if r not in used_rows),
                        key=lambda r: -cand_scores.get((gi, r), 0))
        used   = sorted((r for r in grp if r in used_rows),
                        key=lambda r: -cand_scores.get((gi, r), 0))
        picked = 0
        for r in (unused + used):
            if picked >= count: break
            picks.append(grp[r])
            used_rows.add(r)
            picked += 1
    return frozenset(picks)


def sim_pair_cond(i):
    """PairCond: given current draw, pick 8 numbers with highest co-occurrence with it."""
    current = draw_nums[i]
    cur_arr = np.array(list(current))

    # Score each number: sum of pair_freq with current draw's numbers
    scores = pair_freq[:, cur_arr].sum(axis=1)  # shape (81,)
    scores[0] = -1  # ignore index 0
    for n in current:
        scores[n] = -1  # don't pick numbers in current draw

    picks, used_rows = [], set()
    for _ in range(8):
        best_n, best_score = None, -1
        for n in range(1, 81):
            r = (n-1) // 10
            if r in used_rows: continue
            if scores[n] > best_score:
                best_score = scores[n]
                best_n = n
        if best_n is not None:
            picks.append(best_n)
            used_rows.add((best_n-1) // 10)
            scores[best_n] = -1
    return frozenset(picks)


# ── 5. Run simulation ──────────────────────────────────────────────────────
print(f"\nRunning simulation for {N-1-60} draw-pairs, 3 modes...")
t0 = time.time()

modes = ['baseline', 'pair_delay', 'pair_cond']
res = {m: {'h8':0,'h7':0,'h6':0,'h4':0,'h3':0,'total':0} for m in modes}

for i in range(60, N-1):
    nxt = draw_nums[i+1]

    for mode, fn in [('baseline', sim_nodelay_base),
                     ('pair_delay', sim_pair_delay),
                     ('pair_cond', sim_pair_cond)]:
        ticket = fn(i)
        if len(ticket) < 8: continue
        hits = len(ticket & nxt)
        res[mode]['total'] += 1
        if hits >= 8: res[mode]['h8'] += 1
        if hits >= 7: res[mode]['h7'] += 1
        if hits >= 6: res[mode]['h6'] += 1
        if hits >= 4: res[mode]['h4'] += 1
        if hits >= 3: res[mode]['h3'] += 1

    if (i - 60 + 1) % 50000 == 0:
        print(f"  {i-60+1}/{N-1-60}... {time.time()-t0:.0f}s")

print(f"  Total: {time.time()-t0:.1f}s")

# ── 6. Report ──────────────────────────────────────────────────────────────
print(f"\n{'='*80}")
print(f"{'Mode':<14} {'8/8':>5} {'7/8':>5} {'6/8':>5} {'4/8':>7} {'3/8':>7} {'Total':>8} {'Rate8':>9} {'Rate7':>9}")
print(f"{'-'*80}")
for mode in modes:
    r = res[mode]
    if r['total'] == 0: continue
    print(f"{mode:<14} {r['h8']:>5} {r['h7']:>5} {r['h6']:>5} "
          f"{r['h4']:>7} {r['h3']:>7} {r['total']:>8} "
          f"{r['h8']/r['total']:>9.6f} {r['h7']/r['total']:>9.5f}")

print(f"\nΣΥΜΠΕΡΑΣΜΑ:")
base = res['baseline']
for mode in ['pair_delay', 'pair_cond']:
    r = res[mode]
    d8 = r['h8'] - base['h8']
    d7 = r['h7'] - base['h7']
    print(f"  {mode}: Δ8/8={d8:+d}  Δ7/8={d7:+d}")
