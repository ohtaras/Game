#!/usr/bin/env python3
"""
ΛΕΞΑΡΙΘΜΟΙ (Argyropoulos method) — ανάλυση KINO με ελληνική γεματρία.

Παραδοσιακή ελληνική γεματρία:
  Α=1, Β=2, Γ=3, Δ=4, Ε=5, Ζ=7, Η=8, Θ=9
  Ι=10, Κ=20, Λ=30, Μ=40, Ν=50, Ξ=60, Ο=70, Π=80
  Ρ=100, Σ=200, Τ=300, Υ=400, Φ=500, Χ=600, Ψ=700, Ω=800

Tests:
 1. Direct match: draw sum = γνωστός λεξάριθμος
 2. Πυθαγόρεια αναγωγή (digit-sum reduction, 1-9)
 3. Λεξάριθμος του draw (όλη η ακολουθία ως γράμματα)
 4. Word subset (Argyropoulos: ίσοι λεξάριθμοι = ίσες έννοιες)
 5. Sacred/famous values vs neighborhood frequency
 6. Lexarithm autocorrelation
 7. Συσχέτιση draw sum με Argyropoulos's discovered equivalences
"""
import json, time
import numpy as np
from pathlib import Path
from collections import Counter, defaultdict
from datetime import datetime, timezone, timedelta
from math import sqrt, log, comb
from scipy.stats import norm

DATA_DIR = Path('/home/user/Game/data/raw')
GREEK_ALPHA = 'ΑΒΓΔΕΖΗΘΙΚΛΜΝΞΟΠΡΣΤΥΦΧΨΩ'  # 24 letters
GEMATRIA = {
    'Α': 1, 'Β': 2, 'Γ': 3, 'Δ': 4, 'Ε': 5, 'Ζ': 7, 'Η': 8, 'Θ': 9,
    'Ι': 10, 'Κ': 20, 'Λ': 30, 'Μ': 40, 'Ν': 50, 'Ξ': 60, 'Ο': 70, 'Π': 80,
    'Ρ': 100, 'Σ': 200, 'Τ': 300, 'Υ': 400, 'Φ': 500, 'Χ': 600, 'Ψ': 700, 'Ω': 800
}

def lex(word):
    """Υπολογισμός λεξαρίθμου (αγνοεί τόνους και space)."""
    return sum(GEMATRIA.get(c.upper(), 0) for c in word if c.upper() in GEMATRIA)

# Πυθαγόρεια αναγωγή (έγκλειστος αριθμός)
def reduce_digits(n):
    while n > 9:
        n = sum(int(d) for d in str(n))
    return n

# Curated Greek words/phrases with semantic categories
WORD_DATABASE = {
    # Religious / Sacred
    'religion': [
        'ΘΕΟΣ', 'ΧΡΙΣΤΟΣ', 'ΙΗΣΟΥΣ', 'ΑΓΙΟΣ', 'ΑΓΓΕΛΟΣ', 'ΟΡΘΟΔΟΞΟΣ',
        'ΘΕΟΤΟΚΟΣ', 'ΠΑΝΑΓΙΑ', 'ΧΑΡΙΣΜΑ', 'ΑΜΗΝ', 'ΕΥΧΗ', 'ΨΑΛΜΟΣ',
        'ΣΤΑΥΡΟΣ', 'ΟΥΡΑΝΟΣ', 'ΠΑΡΑΔΕΙΣΟΣ', 'ΑΓΙΑ', 'ΑΓΑΠΗ', 'ΑΓΝΕΙΑ',
        'ΣΩΤΗΡΑΣ', 'ΑΝΑΣΤΑΣΗ', 'ΘΕΙΟΣ', 'ΘΕΙΑ',
    ],
    # Mythology
    'mythology': [
        'ΖΕΥΣ', 'ΗΡΑ', 'ΑΘΗΝΑ', 'ΑΠΟΛΛΩΝ', 'ΑΡΗΣ', 'ΑΡΤΕΜΙΣ',
        'ΑΦΡΟΔΙΤΗ', 'ΕΡΩΣ', 'ΔΙΟΝΥΣΟΣ', 'ΕΡΜΗΣ', 'ΟΡΦΕΥΣ', 'ΟΛΥΜΠΟΣ',
        'ΠΟΣΕΙΔΩΝ', 'ΑΔΗΣ', 'ΠΑΝ', 'ΗΦΑΙΣΤΟΣ', 'ΚΡΟΝΟΣ', 'ΕΣΤΙΑ',
        'ΝΥΞ', 'ΧΑΟΣ', 'ΕΡΕΒΟΣ', 'ΓΑΙΑ', 'ΟΥΡΑΝΟΣ',
    ],
    # Philosophy / Wisdom
    'philosophy': [
        'ΣΟΦΙΑ', 'ΛΟΓΟΣ', 'ΑΛΗΘΕΙΑ', 'ΓΝΩΣΗ', 'ΦΩΣ', 'ΨΥΧΗ',
        'ΑΙΩΝΑΣ', 'ΧΡΟΝΟΣ', 'ΑΡΕΤΗ', 'ΑΙΩΝΙΟΣ', 'ΕΛΕΥΘΕΡΙΑ', 'ΕΙΡΗΝΗ',
        'ΔΗΜΟΚΡΑΤΙΑ', 'ΖΩΗ', 'ΘΑΝΑΤΟΣ', 'ΑΘΑΝΑΣΙΑ', 'ΕΥΔΑΙΜΟΝΙΑ',
        'ΑΡΜΟΝΙΑ', 'ΣΥΜΠΑΝ', 'ΚΟΣΜΟΣ', 'ΑΡΧΗ', 'ΤΕΛΟΣ',
    ],
    # Greek nation/places
    'national': [
        'ΕΛΛΑΣ', 'ΕΛΛΗΝΕΣ', 'ΕΛΛΗΝΑΣ', 'ΑΘΗΝΑ', 'ΣΠΑΡΤΗ',
        'ΑΘΗΝΑΙ', 'ΑΛΕΞΑΝΔΡΟΣ', 'ΟΜΗΡΟΣ', 'ΠΕΡΙΚΛΗΣ',
        'ΠΛΑΤΩΝ', 'ΑΡΙΣΤΟΤΕΛΗΣ', 'ΣΩΚΡΑΤΗΣ', 'ΠΥΘΑΓΟΡΑΣ',
        'ΑΡΓΥΡΟΠΟΥΛΟΣ',
    ],
    # Nature / Cosmos
    'nature': [
        'ΗΛΙΟΣ', 'ΣΕΛΗΝΗ', 'ΑΣΤΗΡ', 'ΟΥΡΑΝΟΣ', 'ΘΑΛΑΣΣΑ',
        'ΑΝΕΜΟΣ', 'ΦΥΣΗ', 'ΓΗ', 'ΠΥΡ', 'ΝΕΡΟ',
        'ΑΕΡΑΣ', 'ΦΩΤΙΑ', 'ΧΩΜΑ', 'ΟΡΟΣ', 'ΠΟΤΑΜΟΣ',
    ],
    # Phrases (with article)
    'phrases': [
        'Η ΑΛΗΘΕΙΑ', 'ΤΟ ΦΩΣ', 'Ο ΘΕΟΣ', 'Η ΖΩΗ',
        'Η ΣΟΦΙΑ', 'Η ΑΓΑΠΗ', 'Ο ΛΟΓΟΣ', 'Η ΨΥΧΗ',
        'ΟΙ ΕΛΛΗΝΕΣ', 'Η ΕΛΛΑΣ', 'Η ΕΙΡΗΝΗ', 'Η ΕΛΕΥΘΕΡΙΑ',
        'ΤΟ ΣΥΜΠΑΝ', 'Ο ΚΟΣΜΟΣ', 'Η ΦΥΣΗ', 'Η ΦΥΣΙΣ',
        'ΕΙΜΑΙ', 'ΕΣΥ ΕΙΣΑΙ', 'ΕΓΩ ΕΙΜΑΙ',
    ],
    # Numbers as words
    'numbers': [
        'ΕΝΑ', 'ΔΥΟ', 'ΤΡΙΑ', 'ΤΕΣΣΕΡΑ', 'ΠΕΝΤΕ', 'ΕΞΙ', 'ΕΠΤΑ',
        'ΟΚΤΩ', 'ΕΝΝΕΑ', 'ΔΕΚΑ', 'ΕΚΑΤΟ', 'ΧΙΛΙΑ',
    ],
}

# Flatten and compute lexarithms
all_words = []
for cat, words in WORD_DATABASE.items():
    for w in words:
        L = lex(w)
        all_words.append((w, L, cat))

print("="*70); print("ΛΕΞΑΡΙΘΜΟΙ — Argyropoulos style"); print("="*70)
print(f"\n  {len(all_words)} λέξεις/φράσεις στη βάση")
print(f"\n  Παραδείγματα λεξαρίθμων:")
example_words = ['ΘΕΟΣ', 'ΧΡΙΣΤΟΣ', 'ΑΓΑΠΗ', 'ΛΟΓΟΣ', 'ΣΟΦΙΑ', 'ΕΛΛΑΣ',
                 'ΑΘΗΝΑ', 'ΟΛΥΜΠΟΣ', 'ΗΛΙΟΣ', 'ΣΕΛΗΝΗ', 'ΨΥΧΗ', 'ΦΩΣ',
                 'ΑΛΗΘΕΙΑ', 'ΓΝΩΣΗ', 'Η ΑΛΗΘΕΙΑ', 'ΟΙ ΕΛΛΗΝΕΣ', 'ΤΟ ΦΩΣ',
                 'ΠΛΑΤΩΝ', 'ΣΩΚΡΑΤΗΣ', 'ΑΡΓΥΡΟΠΟΥΛΟΣ']
for w in example_words:
    print(f"    {w:>16} = {lex(w)}")

# Find words with lexarithm in KINO range (~500-1100)
in_range_words = [(w, L, c) for w, L, c in all_words if 500 <= L <= 1100]
print(f"\n  Λέξεις με λεξάριθμο στο εύρος KINO sums (500-1100): {len(in_range_words)}")

# ═══════════════════════════════════════════════════════════════════
# Load KINO
# ═══════════════════════════════════════════════════════════════════
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

# Compute draw sums
draw_sums = np.array([sum(d) for _, d in all_draws], dtype=np.int32)
mu_sum = draw_sums.mean()
sigma_sum = draw_sums.std()
print(f"  Draw sum: μ={mu_sum:.2f}  σ={sigma_sum:.2f}")

all_signals = []
def add_signal(cat, name, z, det):
    all_signals.append((cat, name, z, det))

# ═══════════════════════════════════════════════════════════════════
# TEST 1: DIRECT MATCH — does draw_sum = lexarithm of famous word more often than expected?
# Each draw_sum has expected frequency ~ N × φ(s; μ, σ)
# Compare observed vs expected (smooth normal approx)
# ═══════════════════════════════════════════════════════════════════
print("\n" + "="*70)
print("[1] DIRECT MATCH: draw_sum == λεξάριθμος")
print("="*70)

# Empirical count of each sum value
sum_counter = Counter(draw_sums.tolist())

# Use empirical neighbors as baseline (more robust than normal fit)
def expected_count_smooth(s, W=20):
    """Average count in [s-W, s+W] excluding s itself."""
    nbrs = [sum_counter.get(v, 0) for v in range(s-W, s+W+1) if v != s]
    if not nbrs: return 0
    return np.mean(nbrs)

print(f"  {'Word/Phrase':>20}  {'Lex':>5}  {'Obs':>5}  {'Exp':>6}  {'z':>6}  {'cat':>10}")
top_matches = []
for w, L, cat in in_range_words:
    obs = sum_counter.get(L, 0)
    exp = expected_count_smooth(L, W=15)
    if exp < 5: continue
    z = (obs - exp) / sqrt(exp)
    top_matches.append((w, L, obs, exp, z, cat))
top_matches.sort(key=lambda x: -abs(x[4]))
for w, L, obs, exp, z, cat in top_matches[:25]:
    flag = " ★" if abs(z) > 3 else (" ✓" if abs(z) > 2 else "")
    print(f"    {w:>20}  {L:>5}  {obs:>5}  {exp:>6.1f}  {z:+6.2f}  {cat:>10}{flag}")
    if abs(z) > 2.5:
        add_signal("lex_match", w, z, f"sum={L} ({w})")

# ═══════════════════════════════════════════════════════════════════
# TEST 2: ΠΥΘΑΓΟΡΕΙΑ ΑΝΑΓΩΓΗ — digital root of draw sums
# ═══════════════════════════════════════════════════════════════════
print("\n" + "="*70)
print("[2] ΠΥΘΑΓΟΡΕΙΑ ΑΝΑΓΩΓΗ (digit reduction → 1-9)")
print("="*70)
reduced = np.array([reduce_digits(int(s)) for s in draw_sums])
print(f"  Expected uniform: ~{N/9:,.0f} draws per digit")
print(f"  {'Digit':>6} {'Count':>10} {'%':>6} {'z':>6}")
exp_per = N / 9
exp_var = exp_per * (1 - 1/9)
for d_ in range(1, 10):
    cnt = int((reduced == d_).sum())
    pct = cnt / N * 100
    z = (cnt - exp_per) / sqrt(exp_var)
    flag = " ★" if abs(z) > 3 else (" ✓" if abs(z) > 2 else "")
    print(f"    {d_:>6} {cnt:>10,} {pct:>5.2f}% {z:+6.2f}{flag}")
    if abs(z) > 2.5:
        add_signal("digit_root", f"d{d_}", z, f"digit root {d_}")

# Digit-root Markov: does reduced sum predict next?
print(f"\n  Πυθαγόρεια Markov chain:")
trans_dr = np.zeros((10, 10), dtype=np.int32)
for i in range(N-1):
    trans_dr[reduced[i], reduced[i+1]] += 1
total_dr = trans_dr[1:, 1:].sum()
chi2 = 0; df = 0
for i in range(1, 10):
    for j in range(1, 10):
        row = trans_dr[i, 1:].sum(); col = trans_dr[1:, j].sum()
        if row * col > 0:
            exp = row * col / total_dr
            if exp > 50:
                chi2 += (trans_dr[i,j] - exp)**2 / exp
                df += 1
chi2_z = (chi2 - df) / sqrt(2*df) if df > 0 else 0
print(f"    chi-square = {chi2:.2f}  df={df}  z={chi2_z:+.2f}")

# ═══════════════════════════════════════════════════════════════════
# TEST 3: GROUPS OF WORDS WITH SAME LEXARITHM (Argyropoulos's "equivalent words")
# ═══════════════════════════════════════════════════════════════════
print("\n" + "="*70)
print("[3] ΟΜΟΨΗΦΕΣ ΛΕΞΕΙΣ — words with same lexarithm")
print("="*70)
lex_groups = defaultdict(list)
for w, L, cat in all_words:
    lex_groups[L].append((w, cat))
shared = [(L, words) for L, words in lex_groups.items() if len(words) >= 2]
shared.sort(key=lambda x: -len(x[1]))
print(f"  Λεξάριθμοι με ≥2 λέξεις: {len(shared)}")
for L, words in shared[:8]:
    obs = sum_counter.get(L, 0)
    exp = expected_count_smooth(L, W=15)
    if exp >= 5:
        z = (obs - exp) / sqrt(exp)
        print(f"    Λ={L}: {' = '.join([w for w,_ in words[:4]])}  obs={obs} (exp {exp:.1f}) z={z:+.2f}")

# ═══════════════════════════════════════════════════════════════════
# TEST 4: LEXARITHM AUTOCORRELATION
# ═══════════════════════════════════════════════════════════════════
print("\n" + "="*70)
print("[4] LEXARITHM AUTOCORRELATION")
print("="*70)
# Treat draw sum as lexarithm-style time series
sums_c = draw_sums.astype(np.float64) - mu_sum
var = np.dot(sums_c, sums_c) / N
print(f"  Draw-sum autocorrelation:")
for lag in [1, 2, 5, 10, 100, 272]:
    cov = np.dot(sums_c[:-lag], sums_c[lag:]) / (N-lag)
    c = cov / var
    z = c * sqrt(N-lag)
    flag = " ★" if abs(z) > 3 else (" ✓" if abs(z) > 2 else "")
    print(f"    lag={lag:>4}: acf={c:+.6f}  z={z:+.2f}{flag}")

# ═══════════════════════════════════════════════════════════════════
# TEST 5: SPECIFIC SACRED VALUES — does any famous lexarithm cluster?
# Check digit-pattern (palindromes, repdigits)
# ═══════════════════════════════════════════════════════════════════
print("\n" + "="*70)
print("[5] ΕΙΔΙΚΕΣ ΑΞΙΕΣ (palindromes, repdigits, sacred)")
print("="*70)
special_values = []
for s in range(500, 1101):
    s_str = str(s)
    if s_str == s_str[::-1]:  # palindrome
        special_values.append((s, 'palindrome'))
    if len(set(s_str)) == 1:  # all same digit
        special_values.append((s, 'repdigit'))
sacred = [
    (666, 'antichrist'), (777, 'sacred trinity'), (888, 'Jesus (ΙΗΣΟΥΣ)'),
    (999, 'completion'), (1000, 'millennium'), (1080, 'cosmic'),
    (700, 'Ψ rare'), (800, 'Ω'), (888, 'octave+'),
]
print(f"  Παλίνδρομοι / Επαναλαμβανόμενοι σε [500-1100]:")
seen = set()
for s, kind in special_values + sacred:
    if s in seen: continue
    seen.add(s)
    obs = sum_counter.get(s, 0)
    exp = expected_count_smooth(s, W=15)
    if exp < 5: continue
    z = (obs - exp) / sqrt(exp)
    flag = " ★" if abs(z) > 3 else (" ✓" if abs(z) > 2 else "")
    print(f"    {s:>4}: obs={obs:>4}  exp={exp:>6.1f}  z={z:+.2f}  ({kind}){flag}")
    if abs(z) > 2.5:
        add_signal("special", f"v{s}", z, f"sum={s} ({kind})")

# ═══════════════════════════════════════════════════════════════════
# TEST 6: LEXARITHM PATTERN — multiples of meaningful numbers
# Famous lexarithm cycles: 7 (sacred), 9 (Pythagorean), 12 (zodiac)
# ═══════════════════════════════════════════════════════════════════
print("\n" + "="*70)
print("[6] DIVISIBILITY: draw sum mod meaningful numbers")
print("="*70)
print(f"  {'Modulus':>8}  {'Description':>30}  {'chi²':>8}  {'z':>6}")
for K, desc in [(7, '7=θεϊκός'), (9, '9=Πυθαγόρειος τέλειος'),
                 (12, '12=ζωδιακός'), (40, '40=Αγία Σαρακοστή'),
                 (108, '108=ιερός'), (144, '144=καρπός Πνεύματος')]:
    residues = draw_sums % K
    cnts = Counter(residues.tolist())
    exp = N / K
    chi2 = sum((cnts.get(r, 0) - exp)**2 / exp for r in range(K))
    df = K - 1
    chi2_z = (chi2 - df) / sqrt(2*df) if df > 0 else 0
    flag = " ★" if abs(chi2_z) > 3 else ""
    print(f"    {K:>8}  {desc:>30}  {chi2:>8.2f}  {chi2_z:+6.2f}{flag}")
    if abs(chi2_z) > 2.5:
        add_signal("mod_sacred", f"mod{K}", chi2_z, f"sum mod {K}")

# ═══════════════════════════════════════════════════════════════════
# TEST 7: WORD-AS-SUBSET (multiset) — same as before but now per-category
# ═══════════════════════════════════════════════════════════════════
print("\n" + "="*70)
print("[7] WORD AS LETTER SUBSET (1-24 mapping)")
print("="*70)
NUM_TO_LET = {i+1: GREEK_ALPHA[i] for i in range(24)}
LET_TO_NUM = {v: k for k, v in NUM_TO_LET.items()}

draw_letter_sets = []
for _, nums in all_draws:
    draw_letter_sets.append(set(n for n in nums if 1 <= n <= 24))

word_results = []
for w, L, cat in all_words:
    # Only works if word uses unique letters in 1-24
    chars = [c for c in w if c.upper() in GREEK_ALPHA]
    if not chars: continue
    nums_needed = set(LET_TO_NUM[c.upper()] for c in chars)
    if len(nums_needed) != len(chars): continue  # repeated letters
    if len(nums_needed) > 10: continue  # too long, P near 0
    # Count
    obs = sum(1 for ls in draw_letter_sets if nums_needed.issubset(ls))
    L_word = len(nums_needed)
    exp_p = comb(80-L_word, 20-L_word) / comb(80, 20)
    exp = N * exp_p
    var = exp * (1 - exp_p)
    if var < 1: continue
    z = (obs - exp) / sqrt(var)
    word_results.append((w, L, L_word, obs, exp, z, cat))
word_results.sort(key=lambda x: -abs(x[5]))
print(f"  Top 15 word subset frequencies (z>|2.5|):")
print(f"  {'Word':>20}  {'L':>3}  {'#let':>4}  {'Obs':>7}  {'Exp':>9}  {'z':>6}")
shown = 0
for w, L, Lw, obs, exp, z, cat in word_results:
    if abs(z) < 2.5: continue
    shown += 1
    if shown > 20: break
    flag = " ★" if abs(z) > 3 else ""
    print(f"    {w:>20}  {L:>3}  {Lw:>4}  {obs:>7}  {exp:>9.1f}  {z:+6.2f}{flag}")
    if abs(z) > 3:
        add_signal("word_subset", w, z, f"word '{w}' subset")

# ═══════════════════════════════════════════════════════════════════
# ΣΥΝΟΨΗ
# ═══════════════════════════════════════════════════════════════════
print("\n" + "="*70)
print(f"ΣΥΝΟΛΟ — {len(all_signals)} σήματα λεξαρίθμων")
print("="*70)
all_signals.sort(key=lambda x: -abs(x[2]))
print(f"\n  Top 15:")
for i, (cat, name, z, det) in enumerate(all_signals[:15], 1):
    print(f"  {i:>3}.  [{cat:>12}]  {name:>20}  z={z:+.2f}  {det}")

cats = defaultdict(list)
for cat, name, z, det in all_signals:
    cats[cat].append(abs(z))
print(f"\n  Σύνοψη ανά κατηγορία:")
for cat in sorted(cats.keys(), key=lambda c: -max(cats[c])):
    arr = cats[cat]
    print(f"    {cat:>14}: {len(arr):>3} σήματα, max |z|={max(arr):.2f}")

out = {'signals': [(c, n, float(z), d) for c, n, z, d in all_signals], 'N': N}
with open('/home/user/Game/argyropoulos_signals.json', 'w') as f:
    json.dump(out, f, indent=2)
print(f"\n  Αποθήκευση: argyropoulos_signals.json")
print("\n" + "="*70)
print("ΤΕΛΟΣ")
print("="*70)
