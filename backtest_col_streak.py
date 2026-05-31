#!/usr/bin/env python3
"""
Approach 2: Column balance — prefer picks that spread across all 10 columns
Approach 3: Hot streak   — within each delay group, prefer numbers that appeared
                           in 2+ of the last 5 draws (recently active)
Both compared against NoTimeBias baseline (delay groups, unused rows first).
"""
import json, time
from pathlib import Path
from collections import deque

DATA_DIR = Path('/home/user/Game/data/raw')
W = 60
DIST_CFG = [3,0,2,1,0,2,0,0,0,0]

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

# ── 2. Precompute delay table ──────────────────────────────────────────────
print("Precomputing delay table...")
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

# ── 3. Precompute recent-5 appearance counts ───────────────────────────────
# recent5[i][n] = how many of draws [i-5..i-1] contained n
print("Precomputing recent-5 appearance counts...")
t0 = time.time()
recent5 = []  # recent5[i] = dict {n: count in last 5}
window = deque(maxlen=5)

for i in range(N):
    counts = {}
    for past_draw in window:
        for n in past_draw:
            counts[n] = counts.get(n, 0) + 1
    recent5.append(counts)
    window.append(draw_nums[i])

print(f"  Done in {time.time()-t0:.1f}s")

# ── 4. Simulation functions ────────────────────────────────────────────────
def sim_baseline(i):
    """Baseline: delay groups, unused rows first (sorted by row index)."""
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


def sim_col_balance(i):
    """
    Column balance: within each delay group, prefer rows whose candidate number
    falls in an unused column. Tie-break by row index.
    """
    entry = by_row_first[i]
    picks, used_rows, used_cols = [], set(), set()
    for gi in range(10):
        count = DIST_CFG[gi]
        if count <= 0: continue
        grp = entry[gi]

        def row_key(r):
            n = grp[r]
            col = (n-1) % 10
            col_used = col in used_cols
            row_used = r in used_rows
            # Sort: unused row + unused col first, then unused row, then used row
            return (int(row_used), int(col_used), r)

        rows = sorted(grp.keys(), key=row_key)
        picked = 0
        for r in rows:
            if picked >= count: break
            n = grp[r]
            picks.append(n)
            used_rows.add(r)
            used_cols.add((n-1) % 10)
            picked += 1
    return frozenset(picks)


def sim_hot_streak(i):
    """
    Hot streak: within each delay group, prefer rows whose candidate number
    appeared in the most of the last 5 draws. Unused rows first still.
    """
    entry = by_row_first[i]
    r5 = recent5[i]  # counts for draws [i-5..i-1]
    picks, used_rows = [], set()
    for gi in range(10):
        count = DIST_CFG[gi]
        if count <= 0: continue
        grp = entry[gi]

        def row_key_streak(r):
            n = grp[r]
            streak = r5.get(n, 0)
            row_used = r in used_rows
            # Prefer unused rows, then higher streak, then lower row index
            return (int(row_used), -streak, r)

        rows = sorted(grp.keys(), key=row_key_streak)
        picked = 0
        for r in rows:
            if picked >= count: break
            picks.append(grp[r])
            used_rows.add(r)
            picked += 1
    return frozenset(picks)


def sim_cold_streak(i):
    """
    Cold streak (inverse): prefer rows whose candidate appeared in FEWEST recent draws.
    'Due' numbers.
    """
    entry = by_row_first[i]
    r5 = recent5[i]
    picks, used_rows = [], set()
    for gi in range(10):
        count = DIST_CFG[gi]
        if count <= 0: continue
        grp = entry[gi]

        def row_key_cold(r):
            n = grp[r]
            streak = r5.get(n, 0)
            row_used = r in used_rows
            return (int(row_used), streak, r)  # lower streak = higher priority

        rows = sorted(grp.keys(), key=row_key_cold)
        picked = 0
        for r in rows:
            if picked >= count: break
            picks.append(grp[r])
            used_rows.add(r)
            picked += 1
    return frozenset(picks)


# ── 5. Run simulation ──────────────────────────────────────────────────────
modes = ['baseline', 'col_balance', 'hot_streak', 'cold_streak']
res = {m: {'h8':0,'h7':0,'h6':0,'h4':0,'h3':0,'total':0} for m in modes}
fns = {'baseline': sim_baseline, 'col_balance': sim_col_balance,
       'hot_streak': sim_hot_streak, 'cold_streak': sim_cold_streak}

print(f"\nRunning simulation for {N-1-60} draw-pairs, {len(modes)} modes...")
t0 = time.time()

for i in range(60, N-1):
    nxt = draw_nums[i+1]
    for mode in modes:
        ticket = fns[mode](i)
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
print(f"\n{'='*85}")
print(f"{'Mode':<14} {'8/8':>5} {'7/8':>5} {'6/8':>5} {'4/8':>7} {'3/8':>7} {'Total':>8} {'Rate8':>10} {'Rate7':>9}")
print(f"{'-'*85}")
for mode in modes:
    r = res[mode]
    if r['total'] == 0: continue
    flag = " ← BASE" if mode == 'baseline' else ""
    print(f"{mode:<14} {r['h8']:>5} {r['h7']:>5} {r['h6']:>5} "
          f"{r['h4']:>7} {r['h3']:>7} {r['total']:>8} "
          f"{r['h8']/r['total']:>10.6f} {r['h7']/r['total']:>9.5f}{flag}")

print(f"\nΣΥΜΠΕΡΑΣΜΑ vs baseline:")
base = res['baseline']
for mode in ['col_balance', 'hot_streak', 'cold_streak']:
    r = res[mode]
    d8 = r['h8'] - base['h8']
    d7 = r['h7'] - base['h7']
    d6 = r['h6'] - base['h6']
    print(f"  {mode:<14}: Δ8/8={d8:+d}  Δ7/8={d7:+d}  Δ6/8={d6:+d}")
