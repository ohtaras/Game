"""
SCRIPT 2: FFT spectral analysis — does any number appear with hidden periodicity?

Method:
  - Build binary matrix M[N, 80]
  - Subtract mean (0.25) → zero-centred
  - FFT each column, sum power spectra across all 80 numbers
  - Under H0 (white noise): E[|X_k|^2] = N * p*(1-p) per number per freq
    → expected total power at freq k = 80 * N * 0.1875
  - Report top 20 frequencies by (total_power / expected) ratio
  - Report z-scores via Gamma approximation:
      Sum of 80 Exp(sigma^2) ~ Gamma(80, sigma^2)
        mean = 80 * exp_one
        std  = sqrt(80) * exp_one
        z ≈ (observed - mean) / std
"""

import json
import glob
import numpy as np

MINS_PER_DRAW = 5.28

# ── Load ──────────────────────────────────────────────────────────────────────
print("Loading draws...")
files = sorted(glob.glob("/home/user/Game/data/raw/kino_raw_*.json"))
draws = []
for f in files:
    with open(f) as fp:
        d = json.load(fp)
    draws.extend(d["draws"])
draws.sort(key=lambda x: x["id"])
N = len(draws)
print(f"  {N:,} draws  |  ID range {draws[0]['id']} – {draws[-1]['id']}")

# ── Binary matrix ─────────────────────────────────────────────────────────────
print("Building matrix...")
M = np.zeros((N, 80), dtype=np.float32)
for i, d in enumerate(draws):
    for n in d["n"]:
        M[i, n - 1] = 1.0

p      = 20.0 / 80.0   # = 0.25
sigma2 = p * (1.0 - p) # = 0.1875

# ── FFT across draw-time axis ─────────────────────────────────────────────────
print("Computing FFT (N×80 matrix)...")
M_centred = M - p                           # zero-centre; shape (N, 80)
F         = np.fft.rfft(M_centred, axis=0)  # shape (N//2+1, 80), complex
power     = np.abs(F) ** 2                  # shape (nfreqs, 80)
nfreqs    = power.shape[0]                  # N//2 + 1

# Total power summed over all 80 numbers per frequency
total_power = power.sum(axis=1)   # shape (nfreqs,)

# ── Expected power under H0 ───────────────────────────────────────────────────
# For DFT of white noise length N with per-sample variance sigma2:
#   E[|X_k|^2] = N * sigma2  for interior frequencies (k != 0, Nyquist)
exp_one   = N * sigma2          # expected power for one number at one freq
exp_total = 80.0 * exp_one      # expected total_power

# ── Gamma approximation for z-score ───────────────────────────────────────────
# total_power[k] is sum of 80 i.i.d. Exp(exp_one) random variables
#   ~ Gamma(80, exp_one)  with mean = 80*exp_one, std = sqrt(80)*exp_one
gamma_mean = 80.0 * exp_one
gamma_std  = np.sqrt(80.0) * exp_one
z_scores   = (total_power - gamma_mean) / gamma_std   # shape (nfreqs,)

ratio = total_power / exp_total   # ratio to expected

# Frequency index → period
freqs_k = np.arange(nfreqs, dtype=np.float64)
with np.errstate(divide="ignore", invalid="ignore"):
    period_draws = np.where(freqs_k > 0, N / freqs_k, np.inf)
period_hours = period_draws * MINS_PER_DRAW / 60.0

# ── Exclude DC (k=0) and Nyquist (k=nfreqs-1) ────────────────────────────────
valid       = np.ones(nfreqs, dtype=bool)
valid[0]    = False
valid[-1]   = False
valid_idx   = np.where(valid)[0]

# Top 20 by total power
top20_order = valid_idx[np.argsort(total_power[valid_idx])[::-1][:20]]

print("\n" + "=" * 80)
print("TOP 20 FREQUENCIES  (highest total power across all 80 numbers)")
print(f"  N={N:,}  per-freq expected per number: {exp_one:.2f}  "
      f"total expected: {exp_total:.2f}")
print(f"  Gamma approx: mean={gamma_mean:.2f}  std={gamma_std:.2f}")
print("=" * 80)
print(f"{'k':>8}  {'Period(draws)':>15}  {'Period(hours)':>14}  "
      f"{'TotalPower':>12}  {'Ratio':>7}  {'z_approx':>9}")
print("-" * 80)
for k in top20_order:
    pd_  = period_draws[k]
    ph   = period_hours[k]
    tp   = total_power[k]
    r    = ratio[k]
    z    = z_scores[k]
    pd_s = f"{pd_:.2f}" if pd_ < 1e6 else "∞"
    ph_s = f"{ph:.4f}" if ph < 1e6 else "∞"
    print(f"{k:>8}  {pd_s:>15}  {ph_s:>14}  {tp:>12.2f}  {r:>7.4f}  {z:>+9.3f}")

# ── Max z-score across all valid frequencies ──────────────────────────────────
max_z_val = float(z_scores[valid].max())
max_z_k   = int(valid_idx[np.argmax(z_scores[valid])])
print(f"\nMax z across {len(valid_idx):,} valid freqs: "
      f"z={max_z_val:+.3f}  k={max_z_k}  "
      f"period={period_draws[max_z_k]:.2f} draws = {period_hours[max_z_k]:.3f} h")

# ── Notable harmonic periods ──────────────────────────────────────────────────
print("\n" + "=" * 80)
print("NOTABLE PERIODIC SIGNALS  (checking specific calendar periods)")
print("=" * 80)
notable = [
    ("1 draw  (~5.28 min)",        1),
    ("2 draws (~10.56 min)",       2),
    ("~1 hour  (11 draws)",        round(60 / MINS_PER_DRAW)),
    ("~3 hours (34 draws)",        round(3 * 60 / MINS_PER_DRAW)),
    ("~6 hours (68 draws)",        round(6 * 60 / MINS_PER_DRAW)),
    ("~12 hours (136 draws)",      round(12 * 60 / MINS_PER_DRAW)),
    ("~24 hours (273 draws)",      round(24 * 60 / MINS_PER_DRAW)),
    ("~48 hours (545 draws)",      round(48 * 60 / MINS_PER_DRAW)),
    ("~7 days (1909 draws)",       round(7 * 24 * 60 / MINS_PER_DRAW)),
    ("~14 days (3818 draws)",      round(14 * 24 * 60 / MINS_PER_DRAW)),
    ("~30 days (8182 draws)",      round(30 * 24 * 60 / MINS_PER_DRAW)),
]
print(f"  {'Period label':35}  {'k':>6}  {'TotalPower':>12}  {'Ratio':>7}  {'z':>8}")
print("-" * 80)
for label, target_period_draws in notable:
    target_k = round(N / target_period_draws)
    target_k = max(1, min(target_k, nfreqs - 2))
    tp = total_power[target_k]
    r  = ratio[target_k]
    z  = z_scores[target_k]
    print(f"  {label:35}  {target_k:>6}  {tp:>12.2f}  {r:>7.4f}  {z:>+8.3f}")

# ── Per-number top frequency (individual periodicity) ────────────────────────
print("\n" + "=" * 80)
print("TOP PERIODICITY PER NUMBER  (strongest individual frequency for each number)")
print(f"  Threshold: z > 3.0 means possibly non-random")
print("=" * 80)

# For each number individually
ind_exp_one = N * sigma2
ind_std     = ind_exp_one   # std of Exp(exp_one) = exp_one  (chi-sq approx z: (X-mu)/sigma)
# Use chi-sq(2) approx: z = (power - exp_one) / exp_one
per_num_power     = power[valid_idx, :]            # (valid_freqs, 80)
per_num_z         = (per_num_power - ind_exp_one) / ind_exp_one  # crude z
per_num_max_z     = per_num_z.max(axis=0)          # (80,)
per_num_peak_k    = valid_idx[per_num_z.argmax(axis=0)]  # (80,)

strong_nums = np.argsort(per_num_max_z)[::-1][:10]
print(f"{'Num':>4}  {'PeakK':>7}  {'Period(d)':>10}  {'Period(h)':>10}  {'z_crude':>9}")
print("-" * 55)
for n in strong_nums:
    k  = per_num_peak_k[n]
    z  = per_num_max_z[n]
    pd = period_draws[k]
    ph = period_hours[k]
    print(f"{n+1:>4}  {k:>7}  {pd:>10.2f}  {ph:>10.4f}  {z:>+9.3f}")

# ── Interpretation ────────────────────────────────────────────────────────────
print("\n" + "=" * 80)
print("INTERPRETATION")
print("=" * 80)
n_freqs_total = len(valid_idx)
# Bonferroni: for N_tests, expect some |z| > 3 by chance
expected_false_positives_3 = n_freqs_total * 2 * 0.00135
expected_false_positives_4 = n_freqs_total * 2 * 0.0000317
print(f"  Valid frequencies tested  : {n_freqs_total:,}")
print(f"  Expected |z|>3 by chance  : {expected_false_positives_3:.1f}")
print(f"  Expected |z|>4 by chance  : {expected_false_positives_4:.3f}")
n_z3 = int((z_scores[valid] > 3.0).sum())
n_z4 = int((z_scores[valid] > 4.0).sum())
print(f"  Observed |z|>3            : {n_z3}")
print(f"  Observed |z|>4            : {n_z4}")
if n_z4 > expected_false_positives_4 * 2:
    print("  >> POSSIBLE PERIODIC SIGNAL detected (z>4 above chance level)")
elif n_z3 > expected_false_positives_3 * 2:
    print("  >> WEAK SIGNAL possible (z>3 above chance level; could be noise)")
else:
    print("  >> No significant periodic signal found. Draws appear spectrally flat.")

print("\nDone.")
