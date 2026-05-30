#!/usr/bin/env python3
"""
Backtest: find best delay-distribution config for 8/8 and 7/8 KINO hits.
Grid: 8 rows x 10 cols, numbers 1-80. Row r = numbers [r*10+1 .. r*10+10].
Algorithm: for each delay group g, pick 'cfg[g]' numbers from different rows
           (prefer rows not yet used across groups).
           Deterministic: within each row, pick the lowest available number.
Prediction targets draw i+1 using delays computed at draw i.
"""
import json, time
from pathlib import Path

DATA_DIR = Path('/home/user/Game/data/raw')
W = 60  # lookback window

# ── 1. Load data ───────────────────────────────────────────────────────────
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
print(f"  {N} draws  ({draw_ids[0]}...{draw_ids[-1]})  in {time.time()-t0:.1f}s")

# ── 2. Precompute: for each draw i, for each delay group g, for each row r,
#       the first (lowest) available number. Stored as by_row_first[i][g][r].
print("Precomputing delay groups and row-first tables...")
t0 = time.time()

last_seen = {n: -W for n in range(1, 81)}
by_row_first = []   # by_row_first[i] = list of 10 dicts: {row: first_number}
group_hits  = [0]*10   # for phase-1 stats
group_count = [0]*10

for i, (draw_id, cur_nums) in enumerate(all_draws):
    entry = [{} for _ in range(10)]
    for n in range(1, 81):
        # delay at draw i (including current draw as delay=0)
        if n in cur_nums:
            d = 0
        else:
            ls = last_seen[n]
            d = i - ls if (i - ls) <= W else W
        g = min(9, d)
        r = (n - 1) // 10
        if r not in entry[g]:          # first (lowest) number in this row×group
            entry[g][r] = n
    by_row_first.append(entry)
    for n in cur_nums:
        last_seen[n] = i

print(f"  Done in {time.time()-t0:.1f}s")

# ── 3. Phase-1: per-group hit rates ────────────────────────────────────────
print("Phase 1: per-group hit rates...")
t0 = time.time()
g_hits  = [0]*10
g_count = [0]*10

for i in range(60, N - 1):
    next_nums = draw_nums[i + 1]
    entry = by_row_first[i]
    seen_in_draw = set()
    for g in range(10):
        for r, n in entry[g].items():
            # Avoid double-counting (each number appears in exactly one group)
            if n not in seen_in_draw:
                seen_in_draw.add(n)
                hit = 1 if n in next_nums else 0
                g_count[g] += 1
                if hit:
                    g_hits[g] += 1

# Actually, count every number (not just first-per-row)
g_hits  = [0]*10
g_count = [0]*10
last_seen2 = {n: -W for n in range(1, 81)}
for i in range(N):
    cur = draw_nums[i]
    if i >= 60 and i < N - 1:
        nxt = draw_nums[i + 1]
        for n in range(1, 81):
            if n in cur:
                d = 0
            else:
                ls = last_seen2[n]
                d = i - ls if (i - ls) <= W else W
            g = min(9, d)
            g_count[g] += 1
            if n in nxt:
                g_hits[g] += 1
    for n in cur:
        last_seen2[n] = i

print(f"  Done in {time.time()-t0:.1f}s")
print(f"\n  {'Group':>8}  {'Count':>9}  {'Hits':>7}  {'Rate':>7}  {'vs 25%':>8}")
for g in range(10):
    if g_count[g]:
        r = g_hits[g] / g_count[g]
        lbl = f"delay {g}" if g < 9 else "delay 9+"
        print(f"  {lbl:>8}  {g_count[g]:>9}  {g_hits[g]:>7}  {r:>7.4f}  {(r-0.25)*100:>+7.3f}%")

hit_rates = [g_hits[g]/g_count[g] if g_count[g] > 0 else 0.25 for g in range(10)]

# ── 4. Generate all configs (sum=8, 0..4 per group, 10 groups) ────────────
print("\nGenerating configs...")
t0 = time.time()

def gen_configs(total=8, n_groups=10, max_per=4):
    configs = []
    def rec(rem, g, cur):
        if g == n_groups:
            if rem == 0:
                configs.append(tuple(cur))
            return
        lo = max(0, rem - max_per * (n_groups - g - 1))
        hi = min(rem, max_per)
        for k in range(lo, hi + 1):
            cur.append(k)
            rec(rem - k, g + 1, cur)
            cur.pop()
    rec(total, 0, [])
    return configs

configs = gen_configs()
print(f"  {len(configs)} configs in {time.time()-t0:.2f}s")

# Score by theoretical P(8/8) = product of per-pick hit rates
scored = []
for cfg in configs:
    prod = 1.0
    exp  = 0.0
    for g in range(10):
        for _ in range(cfg[g]):
            prod *= hit_rates[g]
            exp  += hit_rates[g]
    scored.append((prod, exp, cfg))

scored.sort(reverse=True)
print(f"\n  Top 5 theoretical P(8/8):")
for prod, exp, cfg in scored[:5]:
    print(f"    {list(cfg)} | P(8/8)={prod:.7f} | E[hits]={exp:.4f}")

# ── 5. Simulation on ALL valid draws ───────────────────────────────────────
def sim_config(i, cfg):
    entry = by_row_first[i]   # list of 10 dicts {row: number}
    picks = []
    used_rows = set()
    for g in range(10):
        count = cfg[g]
        if count == 0:
            continue
        grp = entry[g]
        # unused rows first (sorted), then used rows
        rows = sorted(r for r in grp if r not in used_rows) + \
               sorted(r for r in grp if r in used_rows)
        picked = 0
        for r in rows:
            if picked >= count:
                break
            picks.append(grp[r])
            used_rows.add(r)
            picked += 1
    return frozenset(picks)

# Evaluate top 300 configs by theoretical score + baseline
top_cfgs = [cfg for _, _, cfg in scored[:300]]
# Baseline: 1 per each of first 8 groups
top_cfgs.append((1,1,1,1,1,1,1,1,0,0))

total_draws_eval = N - 1 - 60
print(f"\nSimulating {total_draws_eval} draw-pairs for {len(top_cfgs)} configs...")
t0 = time.time()

sim_results = []
for ci, cfg in enumerate(top_cfgs):
    h8 = h7 = h6 = total = 0
    for i in range(60, N - 1):
        ticket = sim_config(i, cfg)
        sz = len(ticket)
        if sz < 8:
            continue
        hits = len(ticket & draw_nums[i + 1])
        total += 1
        if hits >= 8: h8 += 1
        if hits >= 7: h7 += 1
        if hits >= 6: h6 += 1
    if total > 0:
        sim_results.append((h8, h7, h6, total, list(cfg)))
    if (ci + 1) % 50 == 0:
        elapsed = time.time() - t0
        eta = elapsed / (ci + 1) * (len(top_cfgs) - ci - 1)
        print(f"  {ci+1}/{len(top_cfgs)} configs... {elapsed:.0f}s elapsed, ~{eta:.0f}s remaining")

print(f"  Total simulation: {time.time()-t0:.1f}s")

# ── 6. Report ──────────────────────────────────────────────────────────────
# Sort by 8/8 first, then 7/8
sim_results.sort(key=lambda x: (x[0], x[1], x[2]), reverse=True)

print(f"\n{'='*85}")
print("TOP 20 CONFIGS — sorted by 8/8 then 7/8 hits")
print(f"{'='*85}")
print(f"{'Config [d0,d1,d2,d3,d4,d5,d6,d7,d8,d9+]':<46}  {'8/8':>5}  {'7/8':>5}  {'6/8':>5}  {'Total':>7}  {'Rate8':>9}")
print("-"*85)
for h8, h7, h6, total, cfg in sim_results[:20]:
    r8 = h8/total
    flag = " ← ΒΑΣELINE" if list(cfg) == [1,1,1,1,1,1,1,1,0,0] else ""
    print(f"  {str(cfg):<44}  {h8:>5}  {h7:>5}  {h6:>5}  {total:>7}  {r8:>9.6f}{flag}")

# Sort by 7/8 first
sim_results_7 = sorted(sim_results, key=lambda x: (x[1], x[0], x[2]), reverse=True)
print(f"\n{'='*85}")
print("TOP 10 CONFIGS — sorted by 7/8 then 8/8 hits")
print(f"{'='*85}")
print(f"{'Config [d0,d1,d2,d3,d4,d5,d6,d7,d8,d9+]':<46}  {'8/8':>5}  {'7/8':>5}  {'6/8':>5}  {'Total':>7}  {'Rate7':>9}")
print("-"*85)
for h8, h7, h6, total, cfg in sim_results_7[:10]:
    r7 = h7/total
    print(f"  {str(cfg):<44}  {h8:>5}  {h7:>5}  {h6:>5}  {total:>7}  {r7:>9.6f}")

# Overall winner (combined score)
sim_combined = sorted(sim_results, key=lambda x: x[0]*1000 + x[1], reverse=True)
print(f"\n{'='*85}")
print("ΣΥΝΟΛΙΚΟΣ ΝΙΚΗΤΗΣ (8/8 × 1000 + 7/8):")
h8, h7, h6, total, cfg = sim_combined[0]
print(f"  Config: {cfg}")
print(f"  8/8 hits: {h8}  |  7/8 hits: {h7}  |  6/8 hits: {h6}")
print(f"  Out of {total} draw-pairs evaluated")
print(f"  Rate 8/8: {h8/total:.6f}  |  Rate 7/8: {h7/total:.5f}")
