#!/usr/bin/env python3
"""Backtest for 4-spot ticket: find best config for 4/4 and 3/4 hits."""
import json, time
from pathlib import Path

DATA_DIR = Path('/home/user/Game/data/raw')
W = 60

print("Loading draws...")
t0 = time.time()
all_draws = []
for f in sorted(DATA_DIR.glob('kino_raw_*.json')):
    with open(f) as fp:
        data = json.load(fp)
    for d in data.get('draws', []):
        all_draws.append((d['id'], frozenset(d['n'])))
all_draws.sort(key=lambda x: x[0])
draw_ids = [d[0] for d in all_draws]
draw_nums = [d[1] for d in all_draws]
N = len(all_draws)
print(f"  {N} draws in {time.time()-t0:.1f}s")

print("Precomputing by_row_first...")
t0 = time.time()
last_seen = {n: -W for n in range(1, 81)}
by_row_first = []
for i, (draw_id, cur_nums) in enumerate(all_draws):
    entry = [{} for _ in range(10)]
    for n in range(1, 81):
        d = 0 if n in cur_nums else min(W, i - last_seen[n])
        g = min(9, d)
        r = (n - 1) // 10
        if r not in entry[g]:
            entry[g][r] = n
    by_row_first.append(entry)
    for n in cur_nums:
        last_seen[n] = i
print(f"  Done in {time.time()-t0:.1f}s")

def sim_config(i, cfg):
    entry = by_row_first[i]
    picks = []
    used_rows = set()
    for g in range(10):
        count = cfg[g]
        if count == 0: continue
        grp = entry[g]
        rows = sorted(r for r in grp if r not in used_rows) + \
               sorted(r for r in grp if r in used_rows)
        picked = 0
        for r in rows:
            if picked >= count: break
            picks.append(grp[r])
            used_rows.add(r)
            picked += 1
    return frozenset(picks)

# Generate configs summing to 4, max 4 per group, 10 groups
def gen_configs(total, n_groups=10, max_per=4):
    configs = []
    def rec(rem, g, cur):
        if g == n_groups:
            if rem == 0: configs.append(tuple(cur))
            return
        lo = max(0, rem - max_per*(n_groups-g-1))
        hi = min(rem, max_per)
        for k in range(lo, hi+1):
            cur.append(k); rec(rem-k, g+1, cur); cur.pop()
    rec(total, 0, [])
    return configs

configs4 = gen_configs(4)
print(f"Configs for 4-spot: {len(configs4)}")

print(f"Simulating {N-61} draw-pairs for {len(configs4)} configs...")
t0 = time.time()
sim_results = []
for ci, cfg in enumerate(configs4):
    h4 = h3 = total = 0
    for i in range(60, N-1):
        ticket = sim_config(i, cfg)
        if len(ticket) < 4: continue
        hits = len(ticket & draw_nums[i+1])
        total += 1
        if hits >= 4: h4 += 1
        if hits >= 3: h3 += 1
    if total > 0:
        sim_results.append((h4, h3, total, list(cfg)))
    if (ci+1) % 500 == 0:
        print(f"  {ci+1}/{len(configs4)}... {time.time()-t0:.0f}s")

sim_results.sort(key=lambda x: (x[0], x[1]), reverse=True)
print(f"Done in {time.time()-t0:.1f}s")

print(f"\n{'='*80}")
print("TOP 20 CONFIGS for 4-spot — sorted by 4/4 then 3/4")
print(f"{'='*80}")
print(f"{'Config [d0,d1,d2,d3,d4,d5,d6,d7,d8,d9+]':<46}  {'4/4':>6}  {'3/4':>6}  {'Total':>8}  {'Rate4':>9}")
print("-"*80)
for h4, h3, total, cfg in sim_results[:20]:
    print(f"  {str(cfg):<44}  {h4:>6}  {h3:>6}  {total:>8}  {h4/total:>9.6f}")

# Also by 3/4
sim_by3 = sorted(sim_results, key=lambda x: (x[1], x[0]), reverse=True)
print(f"\n{'='*80}")
print("TOP 10 by 3/4 then 4/4")
print(f"{'='*80}")
for h4, h3, total, cfg in sim_by3[:10]:
    print(f"  {str(cfg):<44}  {h4:>6}  {h3:>6}  {total:>8}  {h3/total:>9.5f}")

h4, h3, total, cfg = sim_results[0]
print(f"\nΝΙΚΗΤΗΣ 4/4: {cfg}")
print(f"  4/4: {h4}  |  3/4: {h3}  |  Total: {total}")
