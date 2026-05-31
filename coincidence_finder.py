#!/usr/bin/env python3
"""
Coincidence finder: search for hidden structure in KINO draws.

1. FFT autocorrelation: if RNG has period P, draws i and i+P will overlap more.
   Tests lags 1..50000 efficiently via FFT.
2. Draw sum distribution: is the sum of 20 numbers per draw normally distributed?
   Check for outlier draws with extreme sums.
3. Draw ID → numbers: does the draw ID (sequential integer) predict anything?
   - ID mod 80 vs drawn numbers
   - ID digit sum vs draw sum
4. Compare with a parallel "ideal" random sequence: generate 238K ideal draws,
   compare their statistics to real draws — how indistinguishable are they?
5. Find "twin draws": most similar pair ever in history (highest overlap ≥ 15/20)
"""
import json, time, numpy as np
from pathlib import Path
from math import sqrt, comb
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
        all_draws.append((d['id'], frozenset(d['n'])))

all_draws.sort(key=lambda x: x[0])
draw_ids  = [d[0] for d in all_draws]
draw_nums = [d[1] for d in all_draws]
N = len(all_draws)
print(f"  {N} draws  ids {draw_ids[0]}..{draw_ids[-1]}  in {time.time()-t0:.1f}s")

# Binary matrix M: shape (N, 80), M[i,n-1]=1 if n in draw i
print("Building binary matrix...")
t0 = time.time()
M = np.zeros((N, 80), dtype=np.float32)
for i, draw in enumerate(draw_nums):
    for n in draw:
        M[i, n-1] = 1.0
print(f"  Shape {M.shape}  in {time.time()-t0:.1f}s")

# ── 2. FFT Autocorrelation ─────────────────────────────────────────────────
print("\n── Test 1: FFT Autocorrelation (detect hidden RNG period) ──")
t0 = time.time()

# For each number n, compute autocorrelation of its indicator series
# Sum across all 80 numbers → total overlap at each lag
fft_size = 1
while fft_size < 2*N:
    fft_size *= 2

total_autocorr = np.zeros(fft_size, dtype=np.float64)
for n in range(80):
    series = M[:, n].astype(np.float64)
    F = np.fft.rfft(series, n=fft_size)
    total_autocorr[:fft_size//2+1] += (F * np.conj(F)).real

autocorr = np.fft.irfft(total_autocorr)[:N]
# autocorr[k] = sum over draws of overlap(draw_i, draw_{i+k}) (not normalized)
# Normalize: expected = (20/80)^2 * 80 * (N-k) = 5*(N-k)
mean_overlap = np.zeros(N)
for k in range(N):
    pairs = N - k
    if pairs > 0:
        mean_overlap[k] = autocorr[k] / pairs

print(f"  Done in {time.time()-t0:.1f}s")
print(f"  Expected mean overlap per lag: 5.00")
print(f"  Lag 0 (self): {mean_overlap[0]:.2f}  (expected: 20)")

# Variance of mean overlap at lag k
# Each pair contributes a hypergeometric overlap: E=5, Var=1.688^2=2.848
# Mean overlap at lag k ~ N(5, sqrt(2.848/(N-k)))
lags_to_show = [1, 2, 5, 10, 50, 100, 500, 1000, 5000, 10000]
print(f"\n  {'Lag':>7}  {'Mean overlap':>13}  {'z-score':>9}  {'Flag':>5}")
for k in lags_to_show:
    if k >= N: continue
    pairs = N - k
    var = 2.848 / pairs
    z = (mean_overlap[k] - 5.0) / sqrt(var)
    flag = " ★★★" if abs(z) > 5 else (" ★" if abs(z) > 3 else "")
    print(f"  {k:>7}  {mean_overlap[k]:>13.5f}  {z:>+9.2f}{flag}")

# Find any lag with anomalous mean overlap (top 20 lags by |z|)
print(f"\n  Scanning lags 1..50000 for anomalies...")
max_lags = min(50000, N-1)
pairs_arr = np.arange(max_lags, 0, -1, dtype=np.float64)  # N-1, N-2, ...
# actually pairs at lag k = N-k
pairs_k = np.array([N-k for k in range(1, max_lags+1)], dtype=np.float64)
var_k = 2.848 / pairs_k
z_k = (mean_overlap[1:max_lags+1] - 5.0) / np.sqrt(var_k)

top_idx = np.argsort(np.abs(z_k))[::-1][:20]
print(f"  Top 20 lags by |z| (expected noise, all should be ~3):")
for idx in sorted(top_idx[:10], key=lambda i: -abs(z_k[i])):
    lag = idx + 1
    print(f"    Lag {lag:>6}: mean_overlap={mean_overlap[lag]:.5f}  z={z_k[idx]:+.2f}")

max_z = np.max(np.abs(z_k))
print(f"\n  Max |z| across lags 1..{max_lags}: {max_z:.2f}")
if max_z > 6:
    print(f"  ★★★ SIGNIFICANT PERIODICITY DETECTED!")
elif max_z > 5:
    print(f"  ★ Possible weak periodicity — investigate further")
else:
    print(f"  No significant periodicity. RNG appears non-periodic in this range.")

# ── 3. Draw sum distribution ───────────────────────────────────────────────
print("\n── Test 2: Draw sum distribution ──")
draw_sums = np.array([sum(draw) for draw in draw_nums])
# Expected: E[sum of 20 from 1-80] = 20 * 40.5 = 810
# Var = 20 * Var(one number) * (80-20)/(80-1)
# Var(U[1..80]) = (80^2-1)/12 = 533.25
# Var(sample) = 20 * 533.25 * 60/79 = 8106.67... wait
# Var of sum of hypergeometric sample:
# Var = n * (M/N) * (1-M/N) * (N-n)/(N-1) ... no that's for binary
# For sum: Var = n * sigma^2 * (N-n)/(N-1) where sigma^2=Var(single element)=533.25
# = 20 * 533.25 * 60/79 = 8103.16
E_sum = 20 * 40.5  # 810
Var_sum = 20 * (80**2 - 1) / 12 * 60 / 79
SD_sum = sqrt(Var_sum)
print(f"  Expected sum: {E_sum:.1f}  σ={SD_sum:.2f}")
print(f"  Actual:  mean={draw_sums.mean():.2f}  std={draw_sums.std():.2f}  "
      f"min={draw_sums.min()}  max={draw_sums.max()}")
z_sum = (draw_sums.mean() - E_sum) / (SD_sum / sqrt(N))
print(f"  z-score of mean: {z_sum:+.2f}")

# Extreme draws
high = sorted(enumerate(draw_sums), key=lambda x: -x[1])[:5]
low  = sorted(enumerate(draw_sums), key=lambda x:  x[1])[:5]
print(f"\n  5 highest sum draws:")
for i, s in high:
    z = (s - E_sum) / SD_sum
    print(f"    Draw #{draw_ids[i]} sum={s}  z={z:+.1f}  nums={sorted(draw_nums[i])}")
print(f"\n  5 lowest sum draws:")
for i, s in low:
    z = (s - E_sum) / SD_sum
    print(f"    Draw #{draw_ids[i]} sum={s}  z={z:+.1f}  nums={sorted(draw_nums[i])}")

# ── 4. Draw ID → draw correlation ─────────────────────────────────────────
print("\n── Test 3: Draw ID structure → numbers ──")
# Is (draw_id % 80) + 1 more likely to appear in that draw?
hits_id_mod = 0
for i, draw in enumerate(draw_nums):
    target = (draw_ids[i] % 80) + 1
    if target in draw:
        hits_id_mod += 1
exp_id_mod = N * 0.25
z_id = (hits_id_mod - exp_id_mod) / sqrt(N * 0.25 * 0.75)
print(f"  (draw_id % 80)+1 in draw? Hits={hits_id_mod}  Exp={exp_id_mod:.0f}  z={z_id:+.2f}")

# Digit sum of draw_id vs draw_sum
id_digit_sums = np.array([sum(int(d) for d in str(did)) for did in draw_ids])
corr = np.corrcoef(id_digit_sums, draw_sums)[0,1]
print(f"  Correlation(digit_sum(id), draw_sum) = {corr:+.5f}")

# ── 5. "Twin draws" — most similar pair ───────────────────────────────────
print("\n── Test 4: Twin draws — highest overlap between any two draws ──")
print("  Scanning for draws with overlap ≥ 14/20...")
t0 = time.time()
max_ov = 0
best_pair = None
# Use dot products for efficiency: overlap = M[i] · M[j]
# Scan systematically in chunks
chunk = 1000
top_overlaps = []

for i in range(0, N, chunk):
    chunk_i = M[i:i+chunk]
    # Compare against all j > i+chunk (to avoid self and near-self)
    for j in range(i+chunk, min(i+chunk+5000, N), chunk):
        chunk_j = M[j:j+chunk]
        ov_block = chunk_i @ chunk_j.T  # shape (chunk, chunk)
        max_in_block = ov_block.max()
        if max_in_block >= 14:
            idxs = np.argwhere(ov_block >= 14)
            for ri, rj in idxs:
                top_overlaps.append((int(ov_block[ri,rj]), i+ri, j+rj))

    if i % 10000 == 0:
        print(f"    {i}/{N}... {time.time()-t0:.0f}s")

top_overlaps.sort(reverse=True)
print(f"  Done in {time.time()-t0:.1f}s")
if top_overlaps:
    print(f"\n  Top pairs with highest overlap:")
    for ov, ii, jj in top_overlaps[:10]:
        gap = draw_ids[jj] - draw_ids[ii]
        print(f"    Draw #{draw_ids[ii]} vs #{draw_ids[jj]}: overlap={ov}/20  "
              f"gap={gap} draws apart")
        common = sorted(draw_nums[ii] & draw_nums[jj])
        print(f"      Common: {common}")
else:
    print(f"  No pair with overlap ≥ 14 found")
    # Find the actual max
    print(f"  Finding true maximum overlap (sampling 100K random pairs)...")
    rng = np.random.default_rng(42)
    ii = rng.integers(0, N, 100000)
    jj = rng.integers(0, N, 100000)
    mask = ii != jj
    ii, jj = ii[mask], jj[mask]
    ov = (M[ii] * M[jj]).sum(axis=1)
    print(f"  Max overlap in 100K random pairs: {ov.max():.0f}  "
          f"mean={ov.mean():.3f} (expected 5.0)")

print("\n══════════════════════════════════════════")
print("ΑΠΟΤΕΛΕΣΜΑΤΑ")
print("══════════════════════════════════════════")
