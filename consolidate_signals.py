#!/usr/bin/env python3
"""
Ενοποίηση ΟΛΩΝ των σημάτων από 10 batches σε ένα master αρχείο.

Output: all_signals_master.json — όλα τα σήματα με κατηγορία, z-score, και
χρήσιμα metadata για το ensemble button.
"""
import json
from pathlib import Path
from collections import defaultdict

files = [
    'all_signals.json',           # batch 1 (marginal: hour/dow/month/pair/cols/rows)
    'conditional_signals.json',   # batch 2 (delay→next, n→n, hot, daily, h×d×n, neighbor)
    'more_signals.json',          # batch 3 (2-back, lag, pair→num, triplets)
    'even_more_signals.json',     # batch 4 (3-back, anti-pair, symmetric, torus)
    'final_signals.json',         # batch 5 (component, overlap-markov, pair→pair)
    'sixth_signals.json',         # batch 6 (pair×hour, gap-std, p2n@lag2, partners)
    'seventh_signals.json',       # batch 7 (h×n markov, p2n@lag3+5, dow trans)
    'eighth_signals.json',        # batch 8 (h+pair→n, multi-pair, lunar, dom)
    'ninth_signals.json',         # batch 9 (p2t, cascade, hxn@lag2, loyalty)
    'tenth_signals.json',         # batch 10 (mostly negative confirmations)
]

all_signals = []
for f in files:
    path = Path('/home/user/Game') / f
    if not path.exists():
        print(f"  SKIP (missing): {f}")
        continue
    with open(path) as fp:
        data = json.load(fp)
    sigs = data.get('signals', [])
    print(f"  {f:>30}: {len(sigs)} σήματα")
    for cat, name, z, det in sigs:
        all_signals.append({'batch': f.replace('_signals.json','').replace('all','marginal'),
                            'cat': cat, 'name': name, 'z': z, 'det': det,
                            'abs_z': abs(z)})

print(f"\n  Συνολικά: {len(all_signals)} σήματα")

# Sort by |z|
all_signals.sort(key=lambda x: -x['abs_z'])

# Top 20 strongest signals (excluding tautological sb_pair)
print(f"\n  TOP 20 ΙΣΧΥΡΟΤΕΡΑ ΣΗΜΑΤΑ:")
print(f"  {'#':>4}  {'batch':>15}  {'cat':>14}  {'|z|':>6}  {'sign':>5}  {'name'}")
shown = 0
for s in all_signals:
    if s['cat'] == 'sb_pair': continue  # tautological
    shown += 1
    if shown > 20: break
    sign = '+' if s['z']>0 else '-'
    print(f"  {shown:>4}.  {s['batch']:>15}  {s['cat']:>14}  {s['abs_z']:>6.2f}  {sign:>5}  {s['name']}")

# Category summary
print(f"\n  ΣΥΝΟΨΗ ΑΝΑ ΚΑΤΗΓΟΡΙΑ:")
cats = defaultdict(list)
for s in all_signals:
    if s['cat'] == 'sb_pair': continue
    cats[s['cat']].append(s['abs_z'])
print(f"  {'category':>16}  {'count':>5}  {'max |z|':>8}  {'mean |z|':>8}")
import numpy as np
for cat in sorted(cats.keys(), key=lambda c: -max(cats[c])):
    arr = cats[cat]
    print(f"  {cat:>16}  {len(arr):>5}  {max(arr):>8.2f}  {np.mean(arr):>8.2f}")

print(f"\n  Σύνολο σημάτων (χωρίς tautological): {sum(len(v) for v in cats.values())}")

# Save master
master = {
    'total_count': len(all_signals),
    'real_count': sum(len(v) for v in cats.values()),
    'signals': all_signals,
}
with open('/home/user/Game/all_signals_master.json', 'w') as f:
    json.dump(master, f, indent=2)
print(f"\n  Αποθήκευση: all_signals_master.json")
