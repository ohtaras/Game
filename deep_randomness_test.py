#!/usr/bin/env python3
"""
3 αυστηρά tests για να βρούμε ΟΠΟΙΟΔΗΠΟΤΕ pattern μέσα στο θόρυβο.

Test 1: COMPRESSION RATIO
  Αν υπάρχει οποιαδήποτε δομή, ο gzip/zstd θα τη βρει. Συγκρίνουμε με
  πραγματικά τυχαία δεδομένα ίσου μεγέθους.

Test 2: NIST-STYLE STATISTICAL TESTS
  Tests από τη NIST SP 800-22 σουίτα τυχαιότητας:
   - Frequency (monobit)
   - Runs test
   - Longest run of ones
   - Block frequency
   - Serial correlation (lag 1-100)
   - Approximate entropy

Test 3: NON-LINEAR PATTERN MINING
  Sliding window: για κάθε ακολουθία 5 διαδοχικών draws, υπάρχουν
  pairs αριθμών που εμφανίζονται μόνο μαζί; Αν ναι, αυτό είναι
  conditional pattern που τα γραμμικά tests χάνουν.

Επίσης: Maximum runs (sequence) ανά αριθμό — πόσες φορές διαδοχικά εμφανίστηκε.
"""
import json, time, gzip, bz2, lzma, hashlib
import numpy as np
from pathlib import Path
from collections import Counter
from math import log, sqrt, log2

DATA_DIR = Path('/home/user/Game/data/raw')
print("Loading...")
t0 = time.time()
all_draws = []
for f in sorted(DATA_DIR.glob('kino_raw_*.json')):
    with open(f) as fp:
        data = json.load(fp)
    for d in data.get('draws', []):
        all_draws.append((d['id'], sorted(d['n'])))
all_draws.sort(key=lambda x: x[0])
N = len(all_draws)
print(f"  {N:,} draws  in {time.time()-t0:.1f}s")

# Build raw byte stream from all draws
# Each draw: 20 numbers × 1 byte each = 20 bytes
print("Building byte stream...")
t0 = time.time()
byte_stream = bytearray()
for _, nums in all_draws:
    byte_stream.extend(nums)
print(f"  {len(byte_stream):,} bytes  ({time.time()-t0:.1f}s)")

# ═══════════════════════════════════════════════════════════════════
# TEST 1: COMPRESSION RATIO
# ═══════════════════════════════════════════════════════════════════
print("\n══ TEST 1: Compression test ══")
print("  Αν υπάρχει δομή → οι compressors θα το βρουν")
print("  Πραγματικά τυχαία δεδομένα: ratio ≈ 1.0 (αλλά λόγω alphabet 1-80, λίγο λιγότερο)\n")

# Reference: εντελώς τυχαίο stream ίδιου μήκους από 1-80
rng = np.random.default_rng(42)
random_stream = bytes(rng.integers(1, 81, size=len(byte_stream), dtype=np.uint8).tolist())

# Real RNG reference: AES-CTR-like (best possible randomness)
# We use os.urandom which uses /dev/urandom
import os
crypto_stream = os.urandom(len(byte_stream))
# Reduce to 1-80 range to match alphabet
crypto_arr = np.frombuffer(crypto_stream, dtype=np.uint8)
crypto_arr = (crypto_arr % 80 + 1).astype(np.uint8)
crypto_stream = bytes(crypto_arr.tolist())

print(f"  Method       Original   Compressed   Ratio")
for name, data in [('KINO real',  bytes(byte_stream)),
                   ('Python RNG', random_stream),
                   ('OS urandom', crypto_stream)]:
    for comp_name, comp_fn in [('gzip-9', lambda d: gzip.compress(d, 9)),
                               ('bz2-9',  lambda d: bz2.compress(d, 9)),
                               ('lzma-9', lambda d: lzma.compress(d, preset=9))]:
        compressed = comp_fn(data)
        ratio = len(compressed) / len(data)
        flag = ' ★' if name == 'KINO real' and ratio < 0.95 else ''
        print(f"  {name:>12} {comp_name:>7}: {len(data):>9,} → {len(compressed):>9,}  ratio={ratio:.4f}{flag}")

# ═══════════════════════════════════════════════════════════════════
# TEST 2: NIST-style statistical tests
# Convert each draw to 80-bit binary, concatenate
# ═══════════════════════════════════════════════════════════════════
print("\n══ TEST 2: NIST-style randomness tests on bit stream ══")
print("  Convert each draw to 80 bits (1 if number drawn) → 19,108,400 bits total\n")

# Build bit stream
print("  Building bit stream...")
t0 = time.time()
bit_arr = np.zeros(N * 80, dtype=np.uint8)
for i, (_, nums) in enumerate(all_draws):
    for n in nums:
        bit_arr[i*80 + n-1] = 1
n_bits = len(bit_arr)
n_ones = int(bit_arr.sum())
n_zeros = n_bits - n_ones
print(f"  {n_bits:,} bits ({time.time()-t0:.1f}s)")
print(f"  Ones: {n_ones:,} ({n_ones/n_bits*100:.2f}%)  Zeros: {n_zeros:,} ({n_zeros/n_bits*100:.2f}%)")
print(f"  Expected: 25.00% ones  (20/80 per draw)")

# 2a. Block Frequency Test (blocks of 1000 bits)
print("\n  2a. Block frequency test (blocks of 1,000 bits):")
M = 1000
n_blocks = n_bits // M
block_props = bit_arr[:n_blocks*M].reshape(n_blocks, M).mean(axis=1)
# Chi-square: sum((p - 0.25)^2 / variance)
chi2_block = 4 * M * ((block_props - 0.25)**2).sum()
from scipy import stats
p_block = 1 - stats.chi2.cdf(chi2_block, n_blocks)
print(f"    χ² = {chi2_block:.0f}  df = {n_blocks}  p-value = {p_block:.4f}")
print(f"    {'★ ANOMALOUS' if p_block < 0.001 else 'PASS — uniform across blocks'}")

# 2b. Runs Test
print("\n  2b. Runs test (transitions between 0 and 1):")
runs = 1 + np.sum(bit_arr[1:] != bit_arr[:-1])
# Expected runs: 2*n_bits*p*(1-p) + 1
p = n_ones / n_bits
exp_runs = 2 * n_bits * p * (1-p) + 1
var_runs = 4 * n_bits * p * (1-p) * (1 - 3*p*(1-p))
z_runs = (runs - exp_runs) / sqrt(var_runs)
p_runs = 2 * (1 - stats.norm.cdf(abs(z_runs)))
print(f"    Runs: {runs:,}  Expected: {exp_runs:,.0f}")
print(f"    z = {z_runs:+.2f}  p = {p_runs:.4f}")
print(f"    {'★ ANOMALOUS' if p_runs < 0.001 else 'PASS — runs are random'}")

# 2c. Longest run of ones (in blocks of 10000)
print("\n  2c. Longest run of ones (per draw, max 20 consecutive):")
# Per draw, longest run is bounded. Let's check overall.
max_run = 0; cur_run = 0
for b in bit_arr:
    if b: cur_run += 1; max_run = max(max_run, cur_run)
    else: cur_run = 0
print(f"    Max consecutive 1s anywhere in stream: {max_run}")
# This depends on whether 80th bit of draw_i is 1 and 1st bit of draw_{i+1} is 1
# Theoretical max for 20-from-80: 20 ones in row only if 1..20 all drawn — extremely rare

# 2d. Serial Correlation (lag 1..200)
print("\n  2d. Serial autocorrelation (lag 1..200):")
print("    Computing FFT-based autocorrelation...")
t0 = time.time()
centered = bit_arr.astype(np.float64) - p
fft_size = 1
while fft_size < 2*n_bits: fft_size *= 2
F = np.fft.rfft(centered, n=fft_size)
acf = np.fft.irfft(F * np.conj(F))[:200]
print(f"    Done in {time.time()-t0:.1f}s")

# Normalize: acf[0] = variance × n_bits
acf_norm = acf / acf[0]  # acf_norm[0] = 1
# Significance threshold: z=3 → ±3/sqrt(n_bits)
threshold = 3 / sqrt(n_bits)
sig_lags = [(lag, acf_norm[lag]) for lag in range(1, 200) if abs(acf_norm[lag]) > threshold]
print(f"    Significance threshold (z=3): ±{threshold:.6f}")
print(f"    Lags with |autocorr| > threshold: {len(sig_lags)}/199 (expected ~0.5 by chance)")
if sig_lags:
    print(f"    Top 5 most significant lags:")
    for lag, val in sorted(sig_lags, key=lambda x: -abs(x[1]))[:5]:
        z = val * sqrt(n_bits)
        print(f"      Lag {lag}: autocorr={val:+.6f}  z={z:+.2f}")

# Lag 80 is critical (= one full draw)
print(f"    Critical lag 80 (one draw apart): autocorr={acf_norm[80]:+.6f}  z={acf_norm[80]*sqrt(n_bits):+.2f}")

# 2e. Approximate Entropy
print("\n  2e. Approximate Entropy (m=2 vs m=3):")
# Computed on a sample for speed
sample_size = 100_000
sample = bit_arr[:sample_size]
def phi(m, data):
    counts = Counter()
    for i in range(len(data) - m + 1):
        key = tuple(data[i:i+m])
        counts[key] += 1
    total = sum(counts.values())
    s = 0
    for c in counts.values():
        prob = c/total
        s += prob * log(prob)
    return s
phi_2 = phi(2, sample)
phi_3 = phi(3, sample)
ApEn = phi_2 - phi_3
expected_ApEn = log(2)  # for perfectly random binary
print(f"    ApEn(m=2) on first {sample_size:,} bits: {ApEn:.6f}")
print(f"    Expected for random:                    {expected_ApEn:.6f}")
print(f"    Difference: {abs(ApEn - expected_ApEn)/expected_ApEn*100:.3f}%")

# ═══════════════════════════════════════════════════════════════════
# TEST 3: Maximum streak per number
# ═══════════════════════════════════════════════════════════════════
print("\n══ TEST 3: Maximum consecutive appearances per number ══")
print("  Expected max streak in 238K draws ≈ log(238855)/log(4) = 8.9 draws")

max_streaks = {}
for n in range(1, 81):
    streak = 0; mx = 0
    for _, nums in all_draws:
        if n in nums:
            streak += 1
            if streak > mx: mx = streak
        else:
            streak = 0
    max_streaks[n] = mx

print(f"\n  Top 10 max streaks:")
for n, mx in sorted(max_streaks.items(), key=lambda x: -x[1])[:10]:
    # Expected: geometric, P(streak ≥ k) ≈ 0.25^k × N
    # Probability of streak ≥ mx in N draws ≈ N * 0.25^mx
    prob_or_more = N * (0.25**mx)
    print(f"    Number {n:2d}: max streak = {mx}   (P[≥{mx} in {N:,} draws] ≈ {prob_or_more:.2e})")

print(f"\n  Overall: max streak = {max(max_streaks.values())} (theoretically ~10-12)")

# ═══════════════════════════════════════════════════════════════════
# FINAL VERDICT
# ═══════════════════════════════════════════════════════════════════
print(f"\n══════════════════════════════════════════")
print(f"ΤΕΛΙΚΗ ΕΤΥΜΗΓΟΡΙΑ")
print(f"══════════════════════════════════════════")