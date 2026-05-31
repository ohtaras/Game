#!/usr/bin/env python3
"""
ΣΩΣΤΟ test: σύγκριση KINO με matched-baseline RNG.

Παράγω 238,855 ψεύτικα draws (20 sorted από 80 με python RNG) και
τα τρέχω ακριβώς με τα ίδια tests. Αν το ΠΡΑΓΜΑΤΙΚΟ KINO διαφέρει
στατιστικά από αυτά, ΥΠΑΡΧΕΙ pattern.
"""
import json, time, gzip, bz2, lzma, random
import numpy as np
from pathlib import Path
from math import sqrt, log
from collections import Counter

DATA_DIR = Path('/home/user/Game/data/raw')

print("Loading real KINO...")
real_draws = []
for f in sorted(DATA_DIR.glob('kino_raw_*.json')):
    with open(f) as fp:
        data = json.load(fp)
    for d in data.get('draws', []):
        real_draws.append(sorted(d['n']))
N = len(real_draws)
print(f"  {N:,} real draws")

print("Generating matched-baseline fake draws (20 sorted from 80, Python RNG)...")
random.seed(42)
fake_draws = [sorted(random.sample(range(1, 81), 20)) for _ in range(N)]
print(f"  {N:,} fake draws")

print("\nGenerating crypto-quality fake draws (using SystemRandom = /dev/urandom)...")
sysrand = random.SystemRandom()
crypto_draws = [sorted(sysrand.sample(range(1, 81), 20)) for _ in range(N)]
print(f"  {N:,} crypto draws")

def to_bytes(draws):
    b = bytearray()
    for d in draws: b.extend(d)
    return bytes(b)

def to_bits(draws):
    bits = np.zeros(len(draws) * 80, dtype=np.uint8)
    for i, d in enumerate(draws):
        for n in d:
            bits[i*80 + n-1] = 1
    return bits

# ═══════════════════════════════════════════════════════════════════
# COMPRESSION SHOOTOUT
# ═══════════════════════════════════════════════════════════════════
print("\n══ COMPRESSION TEST (matched baseline) ══")
print(f"  {'Source':>10} {'gzip-9':>10} {'bz2-9':>10} {'lzma-9':>10}")

results = {}
for name, draws in [('KINO', real_draws), ('Fake', fake_draws), ('Crypto', crypto_draws)]:
    data = to_bytes(draws)
    sizes = {
        'gzip': len(gzip.compress(data, 9)),
        'bz2':  len(bz2.compress(data, 9)),
        'lzma': len(lzma.compress(data, preset=9))
    }
    results[name] = sizes
    print(f"  {name:>10}: {sizes['gzip']/len(data):>9.4f} {sizes['bz2']/len(data):>9.4f} {sizes['lzma']/len(data):>9.4f}")

# Compare KINO vs Fake
print(f"\n  KINO vs Fake (RNG):")
for c in ['gzip', 'bz2', 'lzma']:
    diff = results['KINO'][c] - results['Fake'][c]
    diff_pct = diff / results['Fake'][c] * 100
    flag = ' ★★★ KINO COMPRESSES BETTER' if diff < -1000 else (' ★ KINO COMPRESSES MORE' if diff < -100 else ' ✓ identical')
    print(f"    {c:>6}: {diff:+,} bytes ({diff_pct:+.3f}%){flag}")

print(f"\n  KINO vs Crypto (gold standard):")
for c in ['gzip', 'bz2', 'lzma']:
    diff = results['KINO'][c] - results['Crypto'][c]
    diff_pct = diff / results['Crypto'][c] * 100
    flag = ' ★★★ KINO COMPRESSES BETTER' if diff < -1000 else (' ★ KINO COMPRESSES MORE' if diff < -100 else ' ✓ identical')
    print(f"    {c:>6}: {diff:+,} bytes ({diff_pct:+.3f}%){flag}")

# ═══════════════════════════════════════════════════════════════════
# RUNS TEST (on bits) — comparison
# ═══════════════════════════════════════════════════════════════════
print("\n══ RUNS TEST (bit transitions) — matched baseline ══")
for name, draws in [('KINO', real_draws), ('Fake', fake_draws), ('Crypto', crypto_draws)]:
    bits = to_bits(draws)
    n_bits = len(bits)
    runs = 1 + int(np.sum(bits[1:] != bits[:-1]))
    p = bits.mean()
    exp_runs = 2 * n_bits * p * (1-p) + 1
    var = 4 * n_bits * p * (1-p) * (1 - 3*p*(1-p))
    z = (runs - exp_runs) / sqrt(var)
    print(f"  {name:>10}: runs={runs:,}  expected={exp_runs:,.0f}  z={z:+.2f}")

# ═══════════════════════════════════════════════════════════════════
# LAG-80 AUTOCORRELATION — critical test
# Lag 80 = one full draw later. If KINO has draw-to-draw memory, this shows.
# ═══════════════════════════════════════════════════════════════════
print("\n══ LAG AUTOCORRELATION (critical: lag 80 = next draw) ══")
print(f"  {'Source':>10} {'lag 80':>10} {'lag 160':>10} {'lag 800':>10}")
for name, draws in [('KINO', real_draws), ('Fake', fake_draws), ('Crypto', crypto_draws)]:
    bits = to_bits(draws).astype(np.float32)
    p = bits.mean()
    centered = bits - p
    n_bits = len(bits)
    # Compute specific lags directly (avoid full FFT)
    def acf_at(lag):
        return float(np.dot(centered[:-lag], centered[lag:]) / n_bits) / (p*(1-p))
    a80 = acf_at(80)
    a160 = acf_at(160)
    a800 = acf_at(800)
    z80 = a80 * sqrt(n_bits)
    print(f"  {name:>10}: {a80:>+.6f}({z80:+.1f}σ) {a160:>+.6f} {a800:>+.6f}")

# ═══════════════════════════════════════════════════════════════════
# DRAW-LEVEL: hashes of draws, check uniqueness vs theory
# ═══════════════════════════════════════════════════════════════════
print("\n══ DRAW UNIQUENESS — has any draw repeated exactly? ══")
for name, draws in [('KINO', real_draws), ('Fake', fake_draws), ('Crypto', crypto_draws)]:
    seen = set()
    dups = 0
    for d in draws:
        key = tuple(d)
        if key in seen: dups += 1
        seen.add(key)
    print(f"  {name:>10}: {len(draws):,} draws, {len(seen):,} unique, {dups} duplicates")

# ═══════════════════════════════════════════════════════════════════
# DRAW SUM DISTRIBUTION
# ═══════════════════════════════════════════════════════════════════
print("\n══ DRAW SUM DISTRIBUTION ══")
print(f"  Expected: μ=810, σ≈90.0")
for name, draws in [('KINO', real_draws), ('Fake', fake_draws), ('Crypto', crypto_draws)]:
    sums = np.array([sum(d) for d in draws])
    print(f"  {name:>10}: mean={sums.mean():.3f}  std={sums.std():.3f}  "
          f"min={sums.min()}  max={sums.max()}")

# ═══════════════════════════════════════════════════════════════════
# CONSECUTIVE OVERLAP DISTRIBUTION
# ═══════════════════════════════════════════════════════════════════
print("\n══ CONSECUTIVE OVERLAP DISTRIBUTION (hypergeometric expected) ══")
print(f"  {'Source':>10} {'mean':>7} {'std':>7} {'#0':>7} {'#10+':>7} {'max':>5}")
for name, draws in [('KINO', real_draws), ('Fake', fake_draws), ('Crypto', crypto_draws)]:
    sets = [frozenset(d) for d in draws]
    overlaps = np.array([len(sets[i] & sets[i+1]) for i in range(len(sets)-1)])
    print(f"  {name:>10}: {overlaps.mean():>7.3f} {overlaps.std():>7.3f}  "
          f"{(overlaps==0).sum():>7,} {(overlaps>=10).sum():>7,} {overlaps.max():>5}")

print("\n══════════════════════════════════════════")
print("ΣΥΜΠΕΡΑΣΜΑ")
print("══════════════════════════════════════════")
print("Αν το KINO διαφέρει στατιστικά από Fake/Crypto → υπάρχει pattern")
print("Αν είναι ταυτόσημο → είναι αληθινά τυχαίο")