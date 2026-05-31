#!/usr/bin/env python3
"""
Pattern finder: search for real, statistically significant patterns in KINO data.

Tests:
  1. Lagged correlation matrix: P(Y in draw i+k | X in draw i) vs baseline 0.25
     k=1,2,3 — find pairs with highest z-score
  2. Row/column distribution bias per draw
  3. Consecutive repeat rate: P(X in draw i+1 | X in draw i) per number
  4. Bonus number → next draw: does bonus b in draw i appear in draw i+1 main 20?
  5. Draw overlap distribution: how many numbers repeat draw-to-draw?
  6. Sequential pair trigger: X appeared k draws ago → Y more likely now?
"""
import json, time, numpy as np
from pathlib import Path
from math import sqrt
from collections import Counter

DATA_DIR = Path('/home/user/Game/data/raw')

# ── 1. Load ────────────────────────────────────────────────────────────────
print("Loading draws...")
t0 = time.time()
all_draws = []
for f in sorted(DATA_DIR.glob('kino_raw_*.json')):
    with open(f) as fp:
        data = json.load(fp)
    for d in data.get('draws', []):
        all_draws.append((d['id'], frozenset(d['n']), d.get('b', 0)))

all_draws.sort(key=lambda x: x[0])
draw_ids  = [d[0] for d in all_draws]
draw_nums = [d[1] for d in all_draws]
draw_bonus = [d[2] for d in all_draws]
N = len(all_draws)
print(f"  {N} draws  ids {draw_ids[0]}..{draw_ids[-1]}  in {time.time()-t0:.1f}s")

# ── 2. Draw overlap distribution ───────────────────────────────────────────
print("\n── Test 1: Draw-to-draw overlap distribution ──")
overlap_counts = Counter()
for i in range(N-1):
    ov = len(draw_nums[i] & draw_nums[i+1])
    overlap_counts[ov] += 1

# Expected under hypergeometric: E[overlap] = 20*20/80 = 5
E_ov = 20*20/80
V_ov = 20 * 20/80 * (1-20/80) * (80-20)/(80-1)
print(f"  Expected overlap: {E_ov:.2f}  σ={sqrt(V_ov):.3f}")
print(f"  {'Overlap':>8}  {'Count':>8}  {'%':>7}  {'Expected':>10}  {'z':>7}")
total = N - 1
exp_p = {}  # hypergeometric probabilities
from math import comb
for k in range(21):
    exp_p[k] = comb(20,k)*comb(60,20-k)/comb(80,20)
for k in sorted(overlap_counts.keys()):
    obs = overlap_counts[k]
    exp = exp_p.get(k,0) * total
    z = (obs - exp) / sqrt(exp*(1-exp_p.get(k,0))) if exp > 0 else 0
    marker = " ★" if abs(z) > 3 else ""
    print(f"  {k:>8}  {obs:>8}  {obs/total*100:>6.2f}%  {exp:>10.1f}  {z:>+7.2f}{marker}")

# ── 3. Consecutive repeat rate per number ─────────────────────────────────
print("\n── Test 2: P(X in draw i+1 | X in draw i) per number ──")
repeat_given = np.zeros(81)   # count: X in draw i AND X in draw i+1
appear_count = np.zeros(81)   # count: X in draw i

for i in range(N-1):
    for x in draw_nums[i]:
        appear_count[x] += 1
        if x in draw_nums[i+1]:
            repeat_given[x] += 1

baseline_p = 20/80  # 0.25
print(f"  Baseline P(repeat) = {baseline_p}")
print(f"  Number  Appears  Repeats  P(repeat)  z-score")
outliers = []
for x in range(1, 81):
    if appear_count[x] == 0: continue
    p_rep = repeat_given[x] / appear_count[x]
    n = appear_count[x]
    z = (repeat_given[x] - n*baseline_p) / sqrt(n*baseline_p*(1-baseline_p))
    if abs(z) > 2.5:
        outliers.append((z, x, int(appear_count[x]), int(repeat_given[x]), p_rep))

outliers.sort(reverse=True)
print(f"  Numbers with |z|>2.5 (top high):")
for z, x, n, r, p in outliers[:10]:
    r_grid, c_grid = (x-1)//10, (x-1)%10
    print(f"    n={x:3d} (row{r_grid},col{c_grid})  appears={n}  repeats={r}  p={p:.4f}  z={z:+.2f}")
outliers_low = sorted(outliers, key=lambda t: t[0])
print(f"  Numbers with lowest z (cold repeaters):")
for z, x, n, r, p in outliers_low[:5]:
    r_grid, c_grid = (x-1)//10, (x-1)%10
    print(f"    n={x:3d} (row{r_grid},col{c_grid})  appears={n}  repeats={r}  p={p:.4f}  z={z:+.2f}")

# ── 4. Lagged correlation matrix X→Y (lag 1,2,3) ─────────────────────────
print("\n── Test 3: Lagged correlations X in draw i → Y in draw i+k ──")

for lag in [1, 2, 3]:
    print(f"\n  Lag={lag}:")
    # mat[x][y] = count of (x in draw_i AND y in draw_i+lag)
    mat = np.zeros((81, 81), dtype=np.float32)
    for i in range(N - lag):
        xi = np.array(list(draw_nums[i]))
        yi = np.array(list(draw_nums[i+lag]))
        mat[np.ix_(xi, yi)] += 1

    # For each x, baseline: appear_count[x] draws × P(y)=0.25
    # Expected count for pair (x,y) = appear_count[x] * 0.25
    pairs = []
    for x in range(1, 81):
        n_x = mat[x, :].sum() / 20  # approx draws where x appeared
        if n_x < 100: continue
        exp_xy = n_x * baseline_p
        var_xy = n_x * baseline_p * (1 - baseline_p)
        for y in range(1, 81):
            if x == y: continue
            obs = mat[x][y]
            z = (obs - exp_xy) / sqrt(var_xy)
            if abs(z) > 3.5:
                rx, cx = (x-1)//10, (x-1)%10
                ry, cy = (y-1)//10, (y-1)%10
                dist = sqrt((rx-ry)**2 + (cx-cy)**2)
                pairs.append((z, x, y, int(obs), exp_xy, dist))

    pairs.sort(reverse=True)
    print(f"    Top 15 positive (X predicts Y):")
    print(f"    {'X':>4} {'Y':>4} {'Obs':>7} {'Exp':>8} {'z':>7}  {'GridDist':>9}")
    for z, x, y, obs, exp, dist in pairs[:15]:
        print(f"    {x:>4} {y:>4} {obs:>7} {exp:>8.1f} {z:>+7.2f}  {dist:>9.2f}")

    # Check: are the top pairs adjacent on the grid?
    adj_count = sum(1 for z,x,y,obs,exp,dist in pairs[:20] if dist <= 1.5)
    print(f"    Grid-adjacent pairs in top 20: {adj_count}/20")

    pairs_neg = sorted(pairs, key=lambda t: t[0])
    print(f"    Top 5 negative (X predicts NOT Y):")
    for z, x, y, obs, exp, dist in pairs_neg[:5]:
        print(f"    {x:>4} {y:>4} {obs:>7} {exp:>8.1f} {z:>+7.2f}  {dist:>9.2f}")

# ── 5. Bonus number analysis ───────────────────────────────────────────────
print("\n── Test 4: Bonus number predictiveness ──")
bonus_valid = [(draw_bonus[i], draw_nums[i+1]) for i in range(N-1) if draw_bonus[i] > 0]
if bonus_valid:
    bonus_hits = sum(1 for b, nxt in bonus_valid if b in nxt)
    n_bonus = len(bonus_valid)
    exp_hits = n_bonus * baseline_p
    z = (bonus_hits - exp_hits) / sqrt(exp_hits * (1-baseline_p))
    print(f"  Bonus in draw i → appears in draw i+1 main 20:")
    print(f"  Count: {n_bonus}  Hits: {bonus_hits}  Expected: {exp_hits:.0f}  "
          f"Rate: {bonus_hits/n_bonus:.4f}  z={z:+.2f}")
else:
    print("  No bonus data found")

# ── 6. Row & column distribution test ─────────────────────────────────────
print("\n── Test 5: Row/column distribution bias ──")
row_counts = np.zeros(8, dtype=np.int64)
col_counts = np.zeros(10, dtype=np.int64)
total_nums = 0

for draw in draw_nums:
    for n in draw:
        row_counts[(n-1)//10] += 1
        col_counts[(n-1)%10] += 1
        total_nums += 1

exp_row = total_nums / 8
exp_col = total_nums / 10
print(f"  Row distribution (expected ~{exp_row:.0f} each):")
for r in range(8):
    z = (row_counts[r] - exp_row) / sqrt(exp_row)
    marker = " ★" if abs(z) > 3 else ""
    print(f"    Row {r} (nums {r*10+1:2d}-{r*10+10:2d}): {row_counts[r]}  z={z:+.2f}{marker}")

print(f"  Column distribution (expected ~{exp_col:.0f} each):")
for c in range(10):
    z = (col_counts[c] - exp_col) / sqrt(exp_col)
    marker = " ★" if abs(z) > 3 else ""
    print(f"    Col {c}: {col_counts[c]}  z={z:+.2f}{marker}")

# ── 7. Summary ────────────────────────────────────────────────────────────
print("\n══════════════════════════════════════════")
print("ΣΥΝΟΨΗ: Πραγματικά patterns στο KINO;")
print("══════════════════════════════════════════")
