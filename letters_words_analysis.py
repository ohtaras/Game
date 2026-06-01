#!/usr/bin/env python3
"""
ΓΡΑΜΜΑΤΑ & ΛΕΞΕΙΣ — σχέσεις κληρώσεων με ελληνικό αλφάβητο.

Mapping: 1-24 → ΑΒΓΔΕΖΗΘΙΚΛΜΝΞΟΠΡΣΤΥΦΧΨΩ

Tests:
 1. Λέξεις-υπο-σύνολα: για κάθε draw, ποιες λέξεις "περιέχονται" στο multiset
 2. Isopsephy (γεματρία): άθροισμα letter values per draw, ψάχνουμε bias
 3. Vowel/consonant ratio per draw + Markov
 4. Word frequency × hour (συγκεκριμένες λέξεις σε συγκεκριμένες ώρες)
 5. Word "carryover": αν λέξη Χ υπάρχει στο i, ποιες λέξεις προτιμώνται στο i+1
 6. Letter frequencies vs Greek language norm
 7. Vowel pattern transitions
"""
import json, time
import numpy as np
from pathlib import Path
from collections import Counter, defaultdict
from datetime import datetime, timezone, timedelta
from math import sqrt, comb

DATA_DIR = Path('/home/user/Game/data/raw')
GREEK_ALPHA = 'ΑΒΓΔΕΖΗΘΙΚΛΜΝΞΟΠΡΣΤΥΦΧΨΩ'  # 24 letters
VOWELS = set('ΑΕΗΙΟΥΩ')  # 7 vowels
CONSONANTS = set(GREEK_ALPHA) - VOWELS  # 17 consonants

def n_to_letter(n):
    """Map 1..24 → letter. 25..80 → '_'"""
    return GREEK_ALPHA[n-1] if 1 <= n <= 24 else '_'

# Greek words (only with distinct letters, A-Ω uppercase, no repeats)
# Each entry: word that uses only the first 24 letters with no duplicates
GREEK_WORDS = [
    # 2-letter
    'ΑΝ','ΑΣ','ΓΗ','ΕΙ','ΗΣ','ΟΣ','ΟΥ','ΣΕ','ΤΑ','ΤΕ','ΤΗ','ΤΟ','ΩΣ',
    # 3-letter
    'ΑΛΛ','ΑΠΟ','ΕΝΑ','ΕΧΩ','ΗΤΟ','ΗΧΟ','ΘΕΑ','ΘΕΟ','ΘΟΛ','ΘΥΜ','ΙΔΕ','ΙΧΘ',
    'ΚΑΙ','ΚΑΤ','ΛΕΩ','ΜΗΤ','ΝΕΑ','ΝΟΙ','ΟΤΙ','ΠΑΣ','ΠΑΤ','ΠΟΛ','ΠΟΥ','ΡΟΗ',
    'ΣΟΥ','ΣΟΦ','ΣΧΕ','ΤΡΕ','ΥΨΟ','ΦΑΝ','ΦΩΣ','ΧΩΡ','ΧΘΕ','ΩΡΑ','ΩΧΡ',
    'ΟΜΟ','ΟΧΙ','ΝΑΙ','ΕΥΧ','ΑΡΧ','ΓΟΣ','ΖΩΗ','ΘΕΣ','ΑΧΟ','ΑΡΤ','ΟΡΥ',
    # 4-letter
    'ΑΛΦΑ','ΑΥΡΑ','ΒΗΜΑ','ΓΑΛΑ','ΓΑΜΟ','ΓΕΛΩ','ΓΕΡΟ','ΓΗΘΟ','ΓΡΑΨ','ΔΕΜΑ',
    'ΔΕΟΣ','ΔΕΣΗ','ΔΗΛΑ','ΔΙΨΑ','ΔΟΞΑ','ΔΡΟΜ','ΔΥΟΣ','ΕΓΩΙ','ΕΖΗΣ','ΕΡΓΩ',
    'ΖΕΝΙ','ΖΗΛΙ','ΗΛΙΟ','ΗΡΘΕ','ΗΣΚΙ','ΘΕΛΩ','ΘΕΟΣ','ΘΥΜΑ','ΘΩΡΑ','ΙΣΟΣ',
    'ΚΕΡΙ','ΚΟΛΑ','ΛΑΟΣ','ΛΑΧΩ','ΛΕΩΝ','ΛΥΧΝ','ΜΑΘΩ','ΜΕΡΟ','ΜΙΣΟ','ΝΕΟΣ',
    'ΝΟΗΣ','ΝΥΧΙ','ΞΑΝΑ','ΞΕΝΟ','ΞΕΧΩ','ΟΡΓΗ','ΟΣΟΙ','ΟΥΡΑ','ΟΨΗΣ','ΠΕΡΙ',
    'ΠΗΓΕ','ΠΟΘΩ','ΠΡΩΙ','ΡΟΔΑ','ΡΩΓΑ','ΣΟΦΟ','ΣΤΑΝ','ΣΧΟΛ','ΤΑΧΥ','ΤΕΛΟ',
    'ΤΟΠΟ','ΥΜΝΟ','ΦΥΛΟ','ΦΥΣΗ','ΦΩΤΟ','ΧΑΡΑ','ΧΘΕΣ','ΧΟΡΕ','ΧΡΟΝ','ΨΥΧΗ',
    'ΨΥΧΟ','ΩΡΑΙ','ΩΣΑΝ','ΕΘΝΟ','ΕΛΠΙ','ΕΡΩΣ',
    # 5-letter
    'ΑΓΟΡΕ','ΑΔΕΛΦ','ΑΛΛΟΣ','ΑΞΙΟΣ','ΑΡΩΜΑ','ΒΗΧΑΣ','ΓΛΩΣΣ','ΔΕΧΘΩ','ΔΥΝΑΜ',
    'ΖΗΤΩΣ','ΗΛΙΟΣ','ΗΡΩΑΣ','ΘΑΛΑΣ','ΚΑΛΕΣ','ΛΑΜΨΗΣ','ΛΟΓΟΣ','ΜΕΣΟΛ','ΝΟΗΜΑ',
    'ΞΗΡΟΣ','ΠΑΡΕΧ','ΡΟΥΧΟ','ΣΩΜΑΤ','ΤΟΛΜΗ','ΥΨΩΜΑ','ΦΕΓΓΑ','ΦΙΛΗΣ','ΦΛΟΓΑ',
    'ΦΟΡΕΜ','ΧΡΥΣΟ','ΧΩΡΑΣ','ΨΥΧΡΕ','ΩΡΑΙΟ','ΕΡΩΤΗ','ΕΥΡΩΣ',
]
# Filter: only keep words with all distinct letters (multiset → set)
GREEK_WORDS = [w for w in GREEK_WORDS if len(set(w)) == len(w) and all(c in GREEK_ALPHA for c in w)]
GREEK_WORDS = list(set(GREEK_WORDS))  # dedupe
print(f"Loaded {len(GREEK_WORDS)} distinct Greek words (no repeated letters)")

# Convert each word to set of numbers (1..24)
WORD_TO_NUMS = {w: frozenset(GREEK_ALPHA.index(c)+1 for c in w) for w in GREEK_WORDS}

print("\n" + "="*70); print("LOADING"); print("="*70)
t0 = time.time()
all_draws = []
for f in sorted(DATA_DIR.glob('kino_raw_*.json')):
    with open(f) as fp:
        data = json.load(fp)
    for d in data.get('draws', []):
        all_draws.append((d['id'], sorted(d['n'])))
all_draws.sort(key=lambda x: x[0])
N = len(all_draws)
print(f"  {N:,} draws ({time.time()-t0:.1f}s)")

ANCHOR_ID = 1303293
ANCHOR_DT = datetime(2026,5,31,23,55,tzinfo=timezone(timedelta(hours=3)))
def draw_time(idx):
    return ANCHOR_DT + timedelta(minutes=(all_draws[idx][0] - ANCHOR_ID) * 5.28)
hours_arr = np.array([draw_time(i).hour for i in range(N)], dtype=np.int8)

# Pre-compute: which numbers (1..24) are in each draw
print("\nIndexing letter-numbers per draw...")
t0 = time.time()
draw_letter_sets = []
for _, nums in all_draws:
    s = set(n for n in nums if 1 <= n <= 24)
    draw_letter_sets.append(s)
print(f"  {time.time()-t0:.1f}s")
# Mean letter count per draw: 20 * 24/80 = 6.0
print(f"  Mean letters per draw: {np.mean([len(s) for s in draw_letter_sets]):.2f} (exp 6.0)")

all_signals = []
def add_signal(cat, name, z, det):
    all_signals.append((cat, name, z, det))

# ═══════════════════════════════════════════════════════════════════
# TEST 1: WORD CONTAINMENT — count per word, compare to expected
# ═══════════════════════════════════════════════════════════════════
print("\n" + "="*70)
print("[1] WORD-AS-SUBSET FREQUENCY")
print("="*70)
t0 = time.time()
# For each word w (set of L letters), P(all L letters in draw) = ?
# = C(80-L, 20-L) / C(80, 20)
def expected_word_prob(L):
    # P(all L specific numbers in random 20-from-80)
    if L > 20: return 0
    return comb(80-L, 20-L) / comb(80, 20)

word_counts = defaultdict(int)
for letter_set in draw_letter_sets:
    for w, nums in WORD_TO_NUMS.items():
        if nums.issubset(letter_set):
            word_counts[w] += 1

print(f"  Counted in {time.time()-t0:.1f}s")
print(f"  Top 15 word frequencies vs expected:")
word_results = []
for w in GREEK_WORDS:
    L = len(w)
    exp_p = expected_word_prob(L)
    exp_count = N * exp_p
    var_count = exp_count * (1 - exp_p)
    obs = word_counts[w]
    z = (obs - exp_count) / sqrt(max(var_count, 1))
    word_results.append((w, L, obs, exp_count, z))
word_results.sort(key=lambda x: -abs(x[4]))
for w, L, obs, exp, z in word_results[:15]:
    flag = " ★" if abs(z) > 3 else (" ✓" if abs(z) > 2 else "")
    print(f"    {w:>6} (L={L}): obs={obs:>7,} exp={exp:>9.1f}  z={z:+.2f}{flag}")
    if abs(z) > 2.5:
        add_signal("word_freq", w, z, f"word '{w}'")

# ═══════════════════════════════════════════════════════════════════
# TEST 2: ISOPSEPHY (Greek letter values)
# Α=1, Β=2, ..., Ι=10, Κ=20, Λ=30, ..., Ρ=100, Σ=200, ..., Ω=800
# But we use position values (1..24) for mapping. Use traditional gematria.
# ═══════════════════════════════════════════════════════════════════
print("\n" + "="*70)
print("[2] ISOPSEPHY: gematria sum per draw")
print("="*70)
t0 = time.time()
# Traditional Greek isopsephy (no obsolete letters):
GEMATRIA = {
    'Α': 1, 'Β': 2, 'Γ': 3, 'Δ': 4, 'Ε': 5, 'Ζ': 7, 'Η': 8, 'Θ': 9,
    'Ι': 10, 'Κ': 20, 'Λ': 30, 'Μ': 40, 'Ν': 50, 'Ξ': 60, 'Ο': 70, 'Π': 80,
    'Ρ': 100, 'Σ': 200, 'Τ': 300, 'Υ': 400, 'Φ': 500, 'Χ': 600, 'Ψ': 700, 'Ω': 800
}
NUM_TO_GEMATRIA = {i+1: GEMATRIA[c] for i, c in enumerate(GREEK_ALPHA)}

isopsephy_per_draw = np.zeros(N, dtype=np.int32)
for i, ls in enumerate(draw_letter_sets):
    isopsephy_per_draw[i] = sum(NUM_TO_GEMATRIA[n] for n in ls)

print(f"  Mean isopsephy: {isopsephy_per_draw.mean():.2f}")
print(f"  Std isopsephy:  {isopsephy_per_draw.std():.2f}")

# Check anomalies in specific values (e.g., 666, 888)
sacred_values = {
    666: '666 (Beast)',
    777: '777 (Sacred)',
    888: '888 (Christ)',
    1080: '1080 (cosmic)',
    1234: '1234 (sequence)',
}
total = len(isopsephy_per_draw)
# Expected: roughly uniform across a range — use frequency of EACH value vs neighbors
counter_iso = Counter(isopsephy_per_draw.tolist())
print(f"\n  Sacred value frequencies:")
all_values = list(counter_iso.keys())
mean_freq = sum(counter_iso.values()) / len(counter_iso)
for val, name in sacred_values.items():
    cnt = counter_iso.get(val, 0)
    # Compare to neighborhood (±10) average
    neighbors = [counter_iso.get(v, 0) for v in range(val-10, val+11) if v != val]
    nbr_mean = np.mean(neighbors) if neighbors else 0
    nbr_std = np.std(neighbors) if neighbors else 1
    z = (cnt - nbr_mean) / max(nbr_std, 1)
    print(f"    {name}: count={cnt}  neighbors mean={nbr_mean:.1f}±{nbr_std:.1f}  z={z:+.2f}")
    if abs(z) > 3:
        add_signal("isopsephy", f"val{val}", z, f"isopsephy {name}")

# Distribution test: is isopsephy uniform on its range?
print(f"\n  Most-frequent isopsephy values:")
for val, cnt in counter_iso.most_common(10):
    print(f"    {val}: {cnt} draws")

# ═══════════════════════════════════════════════════════════════════
# TEST 3: VOWEL/CONSONANT RATIO per draw
# Vowels are: Α(1), Ε(5), Η(7), Ι(9), Ο(15), Υ(20), Ω(24) — 7 numbers in 1..24
# In the full 1..80 mapping, "vowel positions" = {1, 5, 7, 9, 15, 20, 24}
# Per draw, count how many of these are present
# ═══════════════════════════════════════════════════════════════════
print("\n" + "="*70)
print("[3] VOWEL POSITION COUNTS per draw")
print("="*70)
VOWEL_NUMS = {1, 5, 7, 9, 15, 20, 24}
CONS_NUMS = set(range(1,25)) - VOWEL_NUMS  # 17

# Expected vowels per draw: 20 * 7 / 80 = 1.75
vowel_counts = np.array([len(VOWEL_NUMS & ls) for ls in draw_letter_sets])
exp_vowel_mean = 20 * 7 / 80
exp_vowel_var = 20 * (7/80) * (73/80) * (60/79)
z = (vowel_counts.mean() - exp_vowel_mean) / sqrt(exp_vowel_var/N)
print(f"  Vowel positions ({sorted(VOWEL_NUMS)}): mean={vowel_counts.mean():.4f} (exp {exp_vowel_mean:.4f}, z={z:+.2f})")
if abs(z) > 2: add_signal("vowels", "mean", z, "vowel count mean")

# Markov on vowel count
trans_v = np.zeros((8, 8), dtype=np.int32)
binned = np.clip(vowel_counts, 0, 7)
for i in range(N-1):
    trans_v[binned[i], binned[i+1]] += 1
total_v = trans_v.sum()
row_v = trans_v.sum(axis=1)
col_v = trans_v.sum(axis=0)
chi2 = 0; df = 0
strong = []
for i in range(8):
    for j in range(8):
        if row_v[i] * col_v[j] > 0:
            exp = row_v[i] * col_v[j] / total_v
            if exp > 50:
                z = (trans_v[i,j] - exp) / sqrt(exp)
                chi2 += (trans_v[i,j] - exp)**2 / exp
                df += 1
                if abs(z) > 2.5:
                    strong.append((i, j, trans_v[i,j], z))
chi2_z = (chi2 - df) / sqrt(2*df) if df > 0 else 0
print(f"  Vowel count Markov chi²: {chi2:.2f} df={df} z={chi2_z:+.2f}")

# ═══════════════════════════════════════════════════════════════════
# TEST 4: WORD FREQUENCY × HOUR — do specific words cluster at hours?
# ═══════════════════════════════════════════════════════════════════
print("\n" + "="*70)
print("[4] WORD × HOUR — does word appearance depend on time?")
print("="*70)
t0 = time.time()
# Use top 20 most-frequent words (more samples = more power)
top_words = sorted(GREEK_WORDS, key=lambda w: -word_counts[w])[:20]
word_hour_sigs = []
for w in top_words:
    nums = WORD_TO_NUMS[w]
    # Per-hour count
    for h in range(24):
        h_mask = hours_arr == h
        T = int(h_mask.sum())
        if T < 1000: continue
        # Count how many of these draws contain word
        cnt_h = sum(1 for i in np.where(h_mask)[0] if nums.issubset(draw_letter_sets[i]))
        L = len(nums)
        exp_p = expected_word_prob(L)
        exp_count = T * exp_p
        var_count = exp_count * (1 - exp_p)
        z = (cnt_h - exp_count) / sqrt(max(var_count, 1))
        if abs(z) > 3:
            word_hour_sigs.append((w, h, cnt_h, T, exp_count, z))
word_hour_sigs.sort(key=lambda x: -abs(x[5]))
print(f"  Found {len(word_hour_sigs)} (word, hour) signals  ({time.time()-t0:.1f}s)")
for w, h, cnt, T, exp, z in word_hour_sigs[:10]:
    print(f"    word '{w}' at h={h:>2}: {cnt}/{T} (exp {exp:.1f}) z={z:+.2f}")
    if abs(z) > 3.5:
        add_signal("word_hour", f"{w}@h{h}", z, f"word '{w}' at h={h}")

# ═══════════════════════════════════════════════════════════════════
# TEST 5: WORD CARRYOVER — if word in draw_i, what about draw_{i+1}?
# ═══════════════════════════════════════════════════════════════════
print("\n" + "="*70)
print("[5] WORD CARRYOVER: word at i → word at i+1?")
print("="*70)
t0 = time.time()
top_5_words = sorted(GREEK_WORDS, key=lambda w: -word_counts[w])[:5]
print(f"  Testing top 5 words: {top_5_words}")
carryover_sigs = []
for w1 in top_5_words:
    nums1 = WORD_TO_NUMS[w1]
    # Indices where w1 is subset
    idx_w1 = [i for i, ls in enumerate(draw_letter_sets[:-1]) if nums1.issubset(ls)]
    if len(idx_w1) < 1000: continue
    next_indices = [i+1 for i in idx_w1]
    for w2 in GREEK_WORDS:
        nums2 = WORD_TO_NUMS[w2]
        L = len(nums2)
        # Count occurrences of w2 in those next-draws
        cnt = sum(1 for j in next_indices if nums2.issubset(draw_letter_sets[j]))
        exp_p = expected_word_prob(L)
        exp = len(next_indices) * exp_p
        var = exp * (1 - exp_p)
        z = (cnt - exp) / sqrt(max(var, 1))
        if abs(z) > 4:
            carryover_sigs.append((w1, w2, cnt, len(next_indices), z))
carryover_sigs.sort(key=lambda x: -abs(x[4]))
print(f"  Found {len(carryover_sigs)} word→word carryover signals  ({time.time()-t0:.1f}s)")
for w1, w2, cnt, T, z in carryover_sigs[:10]:
    print(f"    '{w1}' → '{w2}': {cnt}/{T} z={z:+.2f}")
    if abs(z) > 4:
        add_signal("word_co", f"{w1}→{w2}", z, f"'{w1}' → '{w2}'")

# ═══════════════════════════════════════════════════════════════════
# TEST 6: LETTER FREQUENCY vs Greek language norm
# ═══════════════════════════════════════════════════════════════════
print("\n" + "="*70)
print("[6] LETTER FREQUENCY in draws vs Greek language norm")
print("="*70)
# Per-letter (number 1..24) appearances
letter_freq = np.zeros(25, dtype=np.int64)
for ls in draw_letter_sets:
    for n in ls:
        letter_freq[n] += 1
print(f"  Letter frequencies per draw (expected ~{N*20/80:,.0f} each):")
print(f"  {'Letter':>6}  {'Number':>6}  {'Count':>10}  {'%':>6}  {'Greek norm':>10}  {'z':>6}")
# Greek language norms (approximate, %)
GREEK_NORM = {
    'Α': 11.4, 'Ι': 10.2, 'Ο': 9.9, 'Ε': 8.5, 'Τ': 7.8, 'Σ': 7.7,
    'Ν': 6.8, 'Η': 6.0, 'Π': 4.4, 'Υ': 4.1, 'Μ': 3.5, 'Ρ': 3.4,
    'Κ': 3.3, 'Λ': 3.0, 'Ω': 1.8, 'Γ': 1.7, 'Δ': 1.6, 'Χ': 1.2,
    'Θ': 1.1, 'Φ': 0.8, 'Β': 0.6, 'Ξ': 0.2, 'Ζ': 0.2, 'Ψ': 0.2,
}
total_letters = letter_freq[1:25].sum()
for n in range(1, 25):
    letter = GREEK_ALPHA[n-1]
    pct = letter_freq[n] / total_letters * 100
    norm = GREEK_NORM.get(letter, 0)
    diff = pct - 100/24  # vs uniform expectation
    z = diff / sqrt(100/24 * (1 - 1/24) / N)
    print(f"    {letter:>6}  {n:>6}  {letter_freq[n]:>10,}  {pct:>5.2f}%  {norm:>9.2f}%  {z:+.2f}")

# ═══════════════════════════════════════════════════════════════════
# ΣΥΝΟΛΙΚΗ ΚΑΤΑΤΑΞΗ
# ═══════════════════════════════════════════════════════════════════
print("\n" + "="*70)
print(f"ΣΥΝΟΛΙΚΗ ΚΑΤΑΤΑΞΗ — {len(all_signals)} letter/word σήματα")
print("="*70)
all_signals.sort(key=lambda x: -abs(x[2]))
print(f"\n  Top 20:")
for i, (cat, name, z, det) in enumerate(all_signals[:20], 1):
    print(f"  {i:>3}.  [{cat:>12}]  {name:>15}  |z|={abs(z):>6.2f}  {'+' if z>0 else '-'}  {det}")

cats = defaultdict(list)
for cat, name, z, det in all_signals:
    cats[cat].append(abs(z))
print(f"\n  Σύνοψη ανά κατηγορία:")
for cat in sorted(cats.keys(), key=lambda c: -max(cats[c])):
    arr = cats[cat]
    print(f"    {cat:>14}: {len(arr):>3} σήματα, max |z|={max(arr):.2f}")

out = {'signals': [(c, n, float(z), d) for c, n, z, d in all_signals], 'N': N,
       'words_tested': len(GREEK_WORDS)}
with open('/home/user/Game/letters_words_signals.json', 'w') as f:
    json.dump(out, f, indent=2)
print(f"\n  Αποθήκευση: letters_words_signals.json")
print("\n" + "="*70)
print("ΤΕΛΟΣ")
print("="*70)
