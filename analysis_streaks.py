"""
SCRIPT 3: Consecutive streak analysis.

For each number 1-80:
  - Find all consecutive "runs" (sequences of draws where the number appears
    in every draw in a row).
  - Count run lengths globally and per number.
  - Compare to geometric-distribution expectation (p = 20/80 = 0.25).
  - Find the TOP 10 longest individual streaks.
  - Compute coefficient of variation (CV = std/mean) of gap lengths between
    appearances for each number; print most/least regular.
"""

import json
import glob
import numpy as np

# ── Load ──────────────────────────────────────────────────────────────────────
print("Loading draws...")
files = sorted(glob.glob("/home/user/Game/data/raw/kino_raw_*.json"))
draws = []
for f in files:
    with open(f) as fp:
        d = json.load(fp)
    draws.extend(d["draws"])
draws.sort(key=lambda x: x["id"])
N    = len(draws)
ids  = [d["id"] for d in draws]
print(f"  {N:,} draws  |  ID range {ids[0]} – {ids[-1]}")

# ── Binary presence matrix ────────────────────────────────────────────────────
print("Building presence matrix...")
M = np.zeros((N, 80), dtype=np.int8)
for i, d in enumerate(draws):
    for n in d["n"]:
        M[i, n - 1] = 1

p = 20.0 / 80.0   # probability of any number in any draw

# ── Run-length analysis ───────────────────────────────────────────────────────
# For each number, scan the column and record runs.
MAX_RUN  = 10   # track lengths 1..MAX_RUN and "MAX_RUN+" separately
MAX_SHOW = 6    # show lengths 1-6 and 6+ in summary table

# run_counts_per_num[n][k] = number of runs of length k for number n (k=1..MAX_RUN)
run_counts_global = np.zeros(MAX_RUN + 1, dtype=np.int64)   # index 0 unused

# For top-10 longest streaks
all_streaks = []   # list of (length, num, start_draw_idx, end_draw_idx)

# Gap analysis: gaps_per_num[n] = list of gap lengths
gaps_per_num = [[] for _ in range(80)]

print("Analysing runs and gaps...")
for n in range(80):
    col = M[:, n]           # shape (N,)
    positions = np.where(col == 1)[0]   # draw indices where n appears

    # ── Gap analysis ──────────────────────────────────────────────────────
    if len(positions) > 1:
        gaps = np.diff(positions).tolist()
        gaps_per_num[n] = gaps
    else:
        gaps_per_num[n] = []

    # ── Run detection ─────────────────────────────────────────────────────
    # Walk through col and detect consecutive runs of 1s
    run_start  = None
    run_length = 0
    for i in range(N):
        if col[i] == 1:
            if run_start is None:
                run_start  = i
                run_length = 1
            else:
                run_length += 1
        else:
            if run_start is not None:
                # end of run
                all_streaks.append((run_length, n + 1, run_start, run_start + run_length - 1))
                k = min(run_length, MAX_RUN)
                run_counts_global[k] += 1
                run_start  = None
                run_length = 0
    # Handle run that reaches end of data
    if run_start is not None:
        all_streaks.append((run_length, n + 1, run_start, run_start + run_length - 1))
        k = min(run_length, MAX_RUN)
        run_counts_global[k] += 1

# ── Expected run counts ───────────────────────────────────────────────────────
# Under geometric model: a run of length exactly k starts whenever:
#   - position 0, or the previous draw did NOT contain n  (probability 1-p)
#   - and n appears in the next k draws (probability p^k)
#   - and n does NOT appear in draw k+1 (probability 1-p)
# Expected starts of runs of length ≥ k: N * p * p^(k-1) = N * p^k
# Expected runs of length EXACTLY k: N * p^k * (1-p)   [for k >= 1, approx]
# Total expected runs of any length:  N * p * (1-p) + boundary terms ≈ N * p * (1-p) * 80

# Per number, total draws = N, p = 0.25
# Expected runs length exactly k (per number): N * p^k * (1-p)
# Total runs (all numbers): 80 * N * p * (1-p)
def expected_runs_per_num(k, N, p):
    """Expected run count of EXACTLY k for one number.

    A run of exactly length k requires:
      - Position i-1 did NOT contain n  (prob 1-p), OR i=0 (boundary)
      - Positions i..i+k-1 all contain n (prob p^k)
      - Position i+k does NOT contain n (prob 1-p), OR i+k=N (boundary)
    For large N, boundary effects are negligible:
      E[runs of length EXACTLY k] ≈ N * p^k * (1-p)^2
    """
    return N * (p ** k) * ((1 - p) ** 2)

# ── Print global run-length distribution ──────────────────────────────────────
print("\n" + "=" * 74)
print("GLOBAL RUN-LENGTH DISTRIBUTION  (all 80 numbers combined)")
print(f"  N={N:,}  p=20/80=0.25")
print(f"  Expected run-count of length k per number ≈ N * p^k * (1-p)")
print("=" * 74)
print(f"{'RunLen':>7}  {'Observed':>10}  {'Expected':>10}  {'z_score':>9}  {'ratio':>7}")
print("-" * 74)

for k in range(1, MAX_SHOW + 1):
    obs  = int(run_counts_global[k])
    exp  = 80 * expected_runs_per_num(k, N, p)
    # Var of binomial count: exp * (1 - p^k * (1-p)) ≈ exp for large N
    # Use Poisson approx: std ≈ sqrt(exp)
    std  = np.sqrt(max(exp, 1))
    z    = (obs - exp) / std
    ratio = obs / exp if exp > 0 else float("nan")
    label = f"{k}" if k < MAX_SHOW else f"{k}+"
    print(f"{label:>7}  {obs:>10,}  {exp:>10.1f}  {z:>+9.3f}  {ratio:>7.4f}")

# Runs of length >= MAX_SHOW (aggregate)
# P(run len >= k) = p^k * (1-p)  [probability that a run starting at i has length >= k]
# E[runs of length >= MAX_SHOW] ≈ N * p^MAX_SHOW * (1-p)
obs_long = int(run_counts_global[MAX_SHOW:].sum())
exp_long = sum(80 * expected_runs_per_num(k, N, p) for k in range(MAX_SHOW, MAX_RUN + 1))
# Add expected runs > MAX_RUN (tail of geometric)
exp_tail_per_num = (p ** MAX_RUN) * (1 - p) / (1 - p) * p   # P(len > MAX_RUN) = p^MAX_RUN
exp_tail = 80 * N * (p ** (MAX_RUN + 1)) * (1 - p)          # geometric tail
exp_long += exp_tail
print(f"{'≥'+str(MAX_SHOW)+' (all)':>7}  {obs_long:>10,}  {exp_long:>10.1f}")

# ── Top 10 longest individual streaks ────────────────────────────────────────
print("\n" + "=" * 74)
print("TOP 10 LONGEST STREAKS")
print("=" * 74)
all_streaks.sort(key=lambda x: -x[0])
print(f"{'Rank':>4}  {'Num':>4}  {'Length':>7}  {'StartIdx':>9}  "
      f"{'EndIdx':>8}  {'StartID':>9}  {'EndID':>8}")
print("-" * 74)
for rank, (length, num, start_idx, end_idx) in enumerate(all_streaks[:10], 1):
    start_id = ids[start_idx]
    end_id   = ids[end_idx]
    print(f"{rank:>4}  {num:>4}  {length:>7}  {start_idx:>9,}  "
          f"{end_idx:>8,}  {start_id:>9}  {end_id:>8}")

# ── Coefficient of Variation of gaps ─────────────────────────────────────────
print("\n" + "=" * 74)
print("GAP REGULARITY ANALYSIS  (CV = std/mean of inter-appearance gaps)")
print(f"  Under geometric(p=0.25): expected mean gap = 1/p = {1/p:.2f} draws")
print(f"  Expected CV of geometric = sqrt(1-p)/p / (1/p) = sqrt(1-p) ≈ {np.sqrt(1-p):.4f}")
print("=" * 74)

cv_data = []
for n in range(80):
    gaps = gaps_per_num[n]
    if len(gaps) < 2:
        continue
    g   = np.array(gaps, dtype=np.float64)
    m   = g.mean()
    s   = g.std()
    cv  = s / m if m > 0 else float("nan")
    cv_data.append((n + 1, float(m), float(s), float(cv), len(gaps)))

cv_data.sort(key=lambda x: x[3])  # ascending CV = most regular first

print("\nMOST REGULAR numbers (lowest CV, most clock-like appearance pattern):")
print(f"  {'Num':>4}  {'Appearances':>11}  {'MeanGap':>9}  {'StdGap':>9}  {'CV':>8}")
print("-" * 55)
for num, mean_g, std_g, cv, count in cv_data[:10]:
    print(f"  {num:>4}  {count:>11,}  {mean_g:>9.3f}  {std_g:>9.3f}  {cv:>8.5f}")

print("\nMOST IRREGULAR numbers (highest CV, bursty/lumpy appearances):")
print(f"  {'Num':>4}  {'Appearances':>11}  {'MeanGap':>9}  {'StdGap':>9}  {'CV':>8}")
print("-" * 55)
for num, mean_g, std_g, cv, count in cv_data[-10:][::-1]:
    print(f"  {num:>4}  {count:>11,}  {mean_g:>9.3f}  {std_g:>9.3f}  {cv:>8.5f}")

# ── Geometric-distribution expected CV ────────────────────────────────────────
geometric_cv = np.sqrt(1 - p) / 1.0   # since mean=1/p, std=sqrt((1-p)/p^2), cv=sqrt(1-p)
print(f"\n  Expected CV under H0 (geometric): {geometric_cv:.4f}")
mean_cv_obs = np.mean([x[3] for x in cv_data])
std_cv_obs  = np.std([x[3] for x in cv_data])
print(f"  Observed mean CV across 80 nums : {mean_cv_obs:.4f}  ±{std_cv_obs:.4f}")

# ── Per-run-length z-score summary ───────────────────────────────────────────
print("\n" + "=" * 74)
print("STREAK Z-SCORE SUMMARY")
print("=" * 74)
all_z = []
for k in range(1, MAX_RUN + 1):
    obs = int(run_counts_global[k])
    exp = 80 * expected_runs_per_num(k, N, p)
    std = np.sqrt(max(exp, 1))
    z   = (obs - exp) / std
    all_z.append((k, obs, exp, z))
    if abs(z) >= 2.0:
        direction = "MORE" if z > 0 else "FEWER"
        print(f"  Run length {k}: z={z:+.3f}  ({direction} than expected, "
              f"obs={obs:,}  exp={exp:.0f})")

if all(abs(z) < 2.0 for _, _, _, z in all_z):
    print("  All run-length counts within ±2σ of expectation. Looks geometric.")

print("\nDone.")
