#!/usr/bin/env python3
"""
Approach 4: Cold chain — find 8 grid-adjacent numbers with maximum total delay
           (coldest connected cluster on the 8x10 grid)
Approach 5: Hour filter — test if baseline distribution hits better during
           specific hours (03,04,14,15,19,21,22 identified as hot hours)
"""
import json, time
from pathlib import Path
from collections import deque
import calendar

DATA_DIR = Path('/home/user/Game/data/raw')
W = 60
DIST_CFG = [3,0,2,1,0,2,0,0,0,0]
HOT_HOURS = {3, 4, 14, 15, 19, 21, 22}
CALIB_OFFSET = 3  # hours

# Grid adjacency (flat, 8 directions)
def neighbors(n):
    r, c = (n-1)//10, (n-1)%10
    result = []
    for dr in (-1, 0, 1):
        for dc in (-1, 0, 1):
            if dr == 0 and dc == 0: continue
            nr, nc = r+dr, c+dc
            if 0 <= nr <= 7 and 0 <= nc <= 9:
                result.append(nr*10 + nc + 1)
    return result

NEIGHBORS = {n: neighbors(n) for n in range(1, 81)}

# ── 1. Load ────────────────────────────────────────────────────────────────
print("Loading draws...")
t0 = time.time()
all_draws = []
for f in sorted(DATA_DIR.glob('kino_raw_*.json')):
    fname = f.name  # kino_raw_YYYY_MM.json
    parts = fname.replace('kino_raw_','').replace('.json','').split('_')
    year, month = int(parts[0]), int(parts[1])
    with open(f) as fp:
        data = json.load(fp)
    draws_in_file = data.get('draws', [])
    days_in_month = calendar.monthrange(year, month)[1]
    n_draws = len(draws_in_file)
    draws_per_day = n_draws / days_in_month if days_in_month > 0 else 100
    for idx_in_file, d in enumerate(draws_in_file):
        pos_in_day = idx_in_file % max(1, int(draws_per_day))
        est_hour = int(pos_in_day / max(1, draws_per_day) * 24 + CALIB_OFFSET) % 24
        all_draws.append((d['id'], frozenset(d['n']), est_hour))

all_draws.sort(key=lambda x: x[0])
draw_nums = [d[1] for d in all_draws]
draw_hours = [d[2] for d in all_draws]
N = len(all_draws)
print(f"  {N} draws in {time.time()-t0:.1f}s")

# ── 2. Precompute delay table ──────────────────────────────────────────────
print("Precomputing delay table...")
t0 = time.time()
last_seen = {n: -W for n in range(1, 81)}
by_row_first = []
delays_at = []   # full delay vector for each draw (for cold chain)

for i, draw in enumerate(draw_nums):
    delays = {}
    entry = [{} for _ in range(10)]
    for n in range(1, 81):
        d = 0 if n in draw else min(W, i - last_seen[n])
        delays[n] = d
        g = min(9, d)
        r = (n-1) // 10
        if r not in entry[g]:
            entry[g][r] = n
    by_row_first.append(entry)
    delays_at.append(delays)
    for n in draw:
        last_seen[n] = i

print(f"  Done in {time.time()-t0:.1f}s")

# ── 3. Simulation functions ────────────────────────────────────────────────
def sim_baseline(i):
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


def sim_cold_chain(i):
    """
    Greedy BFS from the coldest number on the grid.
    At each step, add the unvisited neighbor with the highest delay.
    Returns 8 grid-adjacent numbers.
    """
    delays = delays_at[i]
    # Start from the coldest number
    start = max(range(1, 81), key=lambda n: delays[n])
    chain = [start]
    in_chain = {start}
    frontier = set(NEIGHBORS[start])

    while len(chain) < 8 and frontier:
        # Pick the frontier number with highest delay
        best = max(frontier, key=lambda n: delays[n])
        chain.append(best)
        in_chain.add(best)
        frontier.discard(best)
        # Add neighbors of best to frontier (if not already in chain)
        for nb in NEIGHBORS[best]:
            if nb not in in_chain:
                frontier.add(nb)

    return frozenset(chain)


def sim_cold_chain_multi(i):
    """
    Try starting from the top-5 coldest numbers, pick the run
    that produces the highest total delay. Returns the best 8.
    """
    delays = delays_at[i]
    sorted_nums = sorted(range(1, 81), key=lambda n: -delays[n])
    best_chain = None
    best_total = -1

    for start in sorted_nums[:5]:
        chain = [start]
        in_chain = {start}
        frontier = set(NEIGHBORS[start])
        while len(chain) < 8 and frontier:
            best = max(frontier, key=lambda n: delays[n])
            chain.append(best)
            in_chain.add(best)
            frontier.discard(best)
            for nb in NEIGHBORS[best]:
                if nb not in in_chain:
                    frontier.add(nb)
        total = sum(delays[n] for n in chain)
        if total > best_total:
            best_total = total
            best_chain = chain

    return frozenset(best_chain) if best_chain else frozenset()


# ── 4. Run simulation ──────────────────────────────────────────────────────
modes = ['baseline', 'cold_chain', 'cold_chain_multi']
res = {m: {'h8':0,'h7':0,'h6':0,'h4':0,'h3':0,'total':0} for m in modes}
# Hour-filtered results for baseline
res_hot  = {'h8':0,'h7':0,'h6':0,'h4':0,'h3':0,'total':0}
res_cold = {'h8':0,'h7':0,'h6':0,'h4':0,'h3':0,'total':0}

fns = {'baseline': sim_baseline,
       'cold_chain': sim_cold_chain,
       'cold_chain_multi': sim_cold_chain_multi}

print(f"\nRunning simulation for {N-1-60} draw-pairs, {len(modes)} modes + hour filter...")
t0 = time.time()

for i in range(60, N-1):
    nxt = draw_nums[i+1]
    hour = draw_hours[i]

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

    # Hour filter on baseline
    ticket_base = fns['baseline'](i)
    if len(ticket_base) >= 8:
        hits = len(ticket_base & nxt)
        bucket = res_hot if hour in HOT_HOURS else res_cold
        bucket['total'] += 1
        if hits >= 8: bucket['h8'] += 1
        if hits >= 7: bucket['h7'] += 1
        if hits >= 6: bucket['h6'] += 1
        if hits >= 4: bucket['h4'] += 1
        if hits >= 3: bucket['h3'] += 1

    if (i - 60 + 1) % 50000 == 0:
        print(f"  {i-60+1}/{N-1-60}... {time.time()-t0:.0f}s")

print(f"  Total: {time.time()-t0:.1f}s")

# ── 5. Report ──────────────────────────────────────────────────────────────
print(f"\n{'='*85}")
print(f"{'Mode':<20} {'8/8':>5} {'7/8':>5} {'6/8':>5} {'4/8':>7} {'Total':>8} {'Rate8':>10} {'Rate7':>9}")
print(f"{'-'*85}")
for mode in modes:
    r = res[mode]
    if r['total'] == 0: continue
    print(f"{mode:<20} {r['h8']:>5} {r['h7']:>5} {r['h6']:>5} "
          f"{r['h4']:>7} {r['total']:>8} "
          f"{r['h8']/r['total']:>10.6f} {r['h7']/r['total']:>9.5f}")

print(f"\n--- Approach 5: Hour filter on baseline ---")
print(f"{'Mode':<22} {'8/8':>5} {'7/8':>5} {'6/8':>5} {'Total':>8} {'Rate8':>10} {'Rate7':>9}")
print(f"{'-'*70}")
for label, r in [('hot hours (3,4,14,15,19,21,22)', res_hot),
                  ('cold hours (rest)', res_cold)]:
    if r['total'] == 0: continue
    print(f"{label:<22} {r['h8']:>5} {r['h7']:>5} {r['h6']:>5} {r['total']:>8} "
          f"{r['h8']/r['total']:>10.6f} {r['h7']/r['total']:>9.5f}")

print(f"\nΣΥΜΠΕΡΑΣΜΑ vs baseline ({res['baseline']['h8']} 8/8 | {res['baseline']['h7']} 7/8):")
base = res['baseline']
for mode in ['cold_chain', 'cold_chain_multi']:
    r = res[mode]
    d8 = r['h8'] - base['h8']
    d7 = r['h7'] - base['h7']
    print(f"  {mode:<22}: Δ8/8={d8:+d}  Δ7/8={d7:+d}")

print(f"\n  hot vs cold hours rate7: "
      f"{res_hot['h7']/res_hot['total'] if res_hot['total'] else 0:.5f} vs "
      f"{res_cold['h7']/res_cold['total'] if res_cold['total'] else 0:.5f}")
