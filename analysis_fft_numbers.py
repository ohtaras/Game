"""
SCRIPT 2: FFT spectral analysis — does any number appear with hidden periodicity?

Method:
  - Build binary matrix M[N, 80]
  - Subtract mean (0.25) → zero-centred
  - FFT each column, sum power spectra across all 80 numbers
  - Under H0 (white noise): E[|X_k|^2] = N * p*(1-p) per number
    → expected total power at freq k = 80 * N * 0.1875
  - Report top 20 frequencies by (total_power / expected) ratio
  - Report z-scores via Gamma approximation:
      For one number: |X_k|^2 ~ Exp(σ²), σ² = N * p * (1-p) / N  [normalised]
      Actually for DFT of length N, real power at freq k (k≠0,N/2):
        E[|X_k|^2] = N * p * (1-p)
      Sum over 80 independent numbers → Gamma(80, N*p*(1-p))
        mean = 80 * N*p*(1-p)
        var  = 80 * (N*p*(1-p))^2
        z ≈ (observed - mean) / std
"""

import json
import glob
import numpy as np

DATA_GLOB = "/home/user/Game/data/raw/kino_raw_*.json"
MINS_PER_DRAW = 5.28

# ── Load ──────────────────────────────────────────────────────────────────────
def load_draws():
    draws = []
    for path in sorted(glob.glob(DATA_GLOB)):
        with open(path) as f:
            obj = json.load(f)
        draws.extend(obj["draws"])
    draws.sort(key=lambda d: d["id"])
    return draws

draws = load_draws()
N = len(draws)
print(f"Loaded {N:,} draws")

# ── Binary matrix ─────────────────────────────────────────────────────────────
M = np.zeros((N, 80), dtype=np.float32)
for i, d in enumerate(draws):
    for n in d["n"]:
        M[i, n - 1] = 1.0

p = 20.0 / 80.0   # = 0.25

# ── FFT ───────────────────────────────────────────────────────────────────────
print("Computing FFT for all 80 numbers …")
# Subtract mean → zero-centred
M_centred = M - p   # shape (N, 80)

# rfft along axis=0 (time axis) → shape (N//2+1, 80)
F = np.fft.rfft(M_centred, axis=0)   # complex128 automatically
power = np.abs(F) ** 2   # shape (nfreqs, 80)

nfreqs = power.shape[0]   # = N//2 + 1

# Total power summed over all 80 numbers, per frequency
total_power = power.sum(axis=1)   # shape (nfreqs,)

# ── Expected power under H0 ───────────────────────────────────────────────────
# For real-valued white noise series of length N with variance σ² = p*(1-p):
#   E[|X_k|^2] = N * σ²  for k = 1 … N/2-1  (one-sided, not DC or Nyquist)
#   (DC and Nyquist are special — we skip them)
sigma2  = p * (1.0 - p)   # = 0.1875
exp_one = N * sigma2       # expected power per number per freq
exp_total = 80.0 * exp_one # expected total_power

# ── Gamma approximation for z-score ───────────────────────────────────────────
# Sum of 80 Exp(exp_one) ~ Gamma(80, exp_one)
# mean = 80 * exp_one, var = 80 * exp_one^2
gamma_mean = 80.0 * exp_one
gamma_std  = np.sqrt(80.0) * exp_one   # std = sqrt(80) * exp_one
z_scores   = (total_power - gamma_mean) / gamma_std   # shape (nfreqs,)

# Ratio
ratio = total_power / exp_total

# Frequency index → period
freqs_k = np.arange(nfreqs)
# freq in cycles/draw
freq_cyc_draw = freqs_k / N
# period in draws (skip k=0 DC)
with np.errstate(divide='ignore', invalid='ignore'):
    period_draws = np.where(freqs_k > 0, N / freqs_k, np.inf)
period_hours = period_draws * MINS_PER_DRAW / 60.0

# ── Skip DC (k=0) and report top 20 ──────────────────────────────────────────
# Exclude DC (k=0) and Nyquist (last)
valid = np.ones(nfreqs, dtype=bool)
valid[0] = False
valid[-1] = False

valid_idx   = np.where(valid)[0]
top20_order = valid_idx[np.argsort(total_power[valid_idx])[::-1][:20]]

print("\n" + "="*80)
print("TOP 20 FREQUENCIES  (by total power across all 80 numbers)")
print("="*80)
print(f"N={N:,}  exp_one={exp_one:.2f}  exp_total={exp_total:.2f}")
print(f"{'k':>8}  {'Period(draws)':>14}  {'Period(hours)':>14}  "
      f"{'TotalPower':>12}  {'Ratio':>7}  {'z_approx':>9}")
print("-"*80)
for k in top20_order:
    pd_ = period_draws[k]
    ph  = period_hours[k]
    tp  = total_power[k]
    r   = ratio[k]
    z   = z_scores[k]
    pd_str = f"{pd_:.1f}" if pd_ < 1e6 else "∞"
    ph_str = f"{ph:.3f}" if ph < 1e6 else "∞"
    print(f"{k:>8}  {pd_str:>14}  {ph_str:>14}  {tp:>12.2f}  {r:>7.4f}  {z:>+9.3f}")

# ── Max z-score across all valid frequencies ──────────────────────────────────
max_z   = z_scores[valid].max()
max_z_k = valid_idx[np.argmax(z_scores[valid])]
print(f"\nMax z-score across all {len(valid_idx):,} valid frequencies: "
      f"z={max_z:+.3f}  at k={max_z_k} "
      f"(period={period_draws[max_z_k]:.1f} draws = {period_hours[max_z_k]:.2f} h)")

# ── Interesting harmonic periods ──────────────────────────────────────────────
print("\n" + "="*80)
print("NOTABLE PERIODS  (closest k to round-number periods)")
print("="*80)
notable = {
    "1 draw (5.28 min)"        : 1,
    "2 draws"                  : 2,
    "1 hour  (~11.4 draws)"    : round(60 / MINS_PER_DRAW),
    "24 hours (~273 draws)"    : round(24 * 60 / MINS_PER_DRAW),
    "7 days  (~1909 draws)"    : round(7 * 24 * 60 / MINS_PER_DRAW),
    "30 days (~8182 draws)"    : round(30 * 24 * 60 / MINS_PER_DRAW),
}
print(f"{'Period label':30}  {'k':>6}  {'TotalPower':>12}  {'Ratio':>7}  {'z':>8}")
print("-"*80)
for label, target_period in notable.items():
    # find the k closest to N/target_period
    target_k = round(N / target_period) if target_period > 0 else 1
    target_k = max(1, min(target_k, nfreqs - 2))
    tp = total_power[target_k]
    r  = ratio[target_k]
    z  = z_scores[target_k]
    print(f"{label:30}  {target_k:>6}  {tp:>12.2f}  {r:>7.4f}  {z:>+8.3f}")

print("\nDone.")
