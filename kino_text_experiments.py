#!/usr/bin/env python3
"""
KINO text experiments — fast version with join() instead of +=
"""
import json, time, random, re
from pathlib import Path
from collections import Counter
from math import comb

DATA_DIR = Path('/home/user/Game/data/raw')

GR_UP  = 'ΑΒΓΔΕΖΗΘΙΚΛΜΝΞΟΠΡΣΤΥΦΧΨΩ'
GR_LOW = 'αβγδεζηθικλμνξοπρστυφχψω'
PUNCT  = ' .,!?;:-()\"«»\n…/0123456789@#&*+='

CHARMAP = [None]
for c in GR_UP:   CHARMAP.append(c)
for c in GR_LOW:  CHARMAP.append(c)
for c in PUNCT:   CHARMAP.append(c)
assert len(CHARMAP) == 81

def n2c(n): return CHARMAP[n] if CHARMAP[n] else ''

print("Loading...")
t0 = time.time()
all_draws = []
for f in sorted(DATA_DIR.glob('kino_raw_*.json')):
    with open(f) as fp:
        data = json.load(fp)
    for d in data.get('draws', []):
        all_draws.append((d['id'], list(d['n'])))
all_draws.sort(key=lambda x: x[0])
N = len(all_draws)
print(f"  {N} draws in {time.time()-t0:.1f}s")

# ══════════════════════════════════════════════════════════════════
# Exp 1: JSON order always sorted → confirmed in previous run
# ══════════════════════════════════════════════════════════════════
print("\n══ Experiment 1: JSON numbers always sorted: YES (100%) ══")
print("  The API returns numbers in ascending order always.")
print("  → Random letter permutations only possible via shuffle.")

# ══════════════════════════════════════════════════════════════════
# Exp 2: Randomly shuffled → word search
# ══════════════════════════════════════════════════════════════════
print("\n══ Experiment 2: Randomly shuffled stream → word search ══")
random.seed(42)
t0 = time.time()
# Fast: build list then join
parts = []
for _, nums in all_draws:
    s = nums[:]
    random.shuffle(s)
    parts.append(''.join(n2c(n) for n in s))
shuffled_stream = ''.join(parts)
print(f"  Stream: {len(shuffled_stream):,} chars in {time.time()-t0:.1f}s")

GREEK_WORDS = [
    'ΚΑΙ','ΤΟΥ','ΤΗΝ','ΤΟΝ','ΓΙΑ','ΑΠΟ','ΣΤΟ',
    'ΕΝΑ','ΜΙΑ','ΔΕΝ','ΝΑΙ','ΟΧΙ',
    'ΚΙΝΟ','ΝΙΚΗ','ΤΥΧΗ','ΘΕΑ','ΖΩΗ',
    'ΘΕΟ','ΕΛΛΑ',
    'ΚΑΙ','ΤΑ','ΤΟ','ΜΕ','ΝΑ',
    'ΓΗ','ΦΩΣ','ΩΔΗ',
]

print(f"\n  {'Λέξη':>12} {'Βρέθηκε':>9} {'Αναμενόμ':>11} {'Ratio':>7}")
for word in sorted(set(GREEK_WORDS), key=len):
    count = shuffled_stream.count(word)
    # P(specific sequence of k chars in shuffled draw)
    # Each position in a shuffled 20-draw has P(char c) = count(c in 1..80)/80
    # Simplified: uniform P(any letter) = 48/80 = 0.6, P(specific uppercase) = 1/80, etc.
    p = (1/80) ** len(word)
    expected = len(shuffled_stream) * p
    ratio = count/expected if expected > 0.01 else float('inf')
    flag = ' ★' if count > expected * 2 else ''
    print(f"  {word:>12} {count:>9,} {expected:>11.1f} {ratio:>7.2f}{flag}")

# Top 3-letter Greek sequences found
gr_only = re.sub(r'[^ΑΒΓΔΕΖΗΘΙΚΛΜΝΞΟΠΡΣΤΥΦΧΨΩαβγδεζηθικλμνξοπρστυφχψω]', '', shuffled_stream)
tris = Counter(gr_only[i:i+3] for i in range(len(gr_only)-2))
print(f"\n  Top 20 τριγράμματα (shuffled, Greek only):")
for tri, cnt in tris.most_common(20):
    print(f"    '{tri}': {cnt:,}")

# ══════════════════════════════════════════════════════════════════
# Exp 3: Draws containing specific letter subsets (subset = word)
# ══════════════════════════════════════════════════════════════════
print("\n══ Experiment 3: Draws containing letters for Greek words ══")

word_nums = {
    'ΚΑΙ':  frozenset([10, 1, 9]),    # Κ=10,Α=1,Ι=9
    'ΤΗΝ':  frozenset([19, 7, 13]),
    'ΕΙΝ':  frozenset([5, 9, 13]),
    'ΝΑΙ':  frozenset([13, 1, 9]),
    'ΓΗ':   frozenset([3, 7]),
    'ΘΕΑ':  frozenset([8, 5, 1]),
    'ΖΩΗ':  frozenset([6, 24, 7]),
    'ΝΙΚΗ': frozenset([13, 9, 10, 7]),
    'ΚΙΝΟ': frozenset([10, 9, 13, 15]),  # Κ=10,Ι=9,Ν=13,Ο=15
    'ΤΥΧΗ': frozenset([19, 20, 22, 7]),  # Τ=19,Υ=20,Χ=22,Η=7
    'ΤΥΧΗ_lo': frozenset([19, 20, 22, 7]),
    'ΝΙΚΩ': frozenset([13, 9, 10, 24]),  # Ν=13,Ι=9,Κ=10,Ω=24
    'ΕΛΛΑ': frozenset([5, 12, 12, 1]),   # Ε=5,Λ=12,Λ=12,Α=1 — has duplicate!
}

draw_sets = [frozenset(n) for _, n in all_draws]
for word, needed in word_nums.items():
    if word == 'ΕΛΛΑ': continue  # skip duplicate
    count = sum(1 for ds in draw_sets if needed <= ds)
    k = len(needed)
    expected = N * comb(80-k, 20-k) / comb(80, 20)
    print(f"  '{word}' [{','.join(str(x) for x in sorted(needed))}]: "
          f"{count:,}× βρέθηκε | αναμένεται {expected:.0f}× | ratio={count/expected:.2f}")

# ══════════════════════════════════════════════════════════════════
# Exp 4: "Bible code" — every K-th draw, max(nums) as letter
# ══════════════════════════════════════════════════════════════════
print("\n══ Experiment 4: 'Bible code' — Kth draw, first number ══")
for K in [1, 5, 10, 20]:
    code = [n2c(all_draws[i][1][0]) for i in range(0, min(300*K, N), K)]
    s = ''.join(c for c in code if c and c.strip() and c.isalpha())
    # Look for common Greek words in the code
    found = []
    for w in ['ΚΑΙ','ΓΗ','ΝΑΙ','ΤΟ','ΝΑ','ΜΕ','ΕΝΑ','ΘΕΑ','ΝΙΚΗ','ΤΥΧΗ']:
        if w in s:
            found.append(w)
    print(f"  K={K:3d}: {s[:80]}{'...' if len(s)>80 else ''}")
    print(f"         Words found: {found if found else 'none'}")

# ══════════════════════════════════════════════════════════════════
# Exp 5: Frequency-based mapping (most drawn → most common Greek letter)
# ══════════════════════════════════════════════════════════════════
print("\n══ Experiment 5: Frequency-based custom mapping ══")

num_freq = Counter()
for _, nums in all_draws:
    for n in nums:
        num_freq[n] += 1

# Greek letter frequency order (approximate, Uppercase only for simplicity)
GREEK_FREQ = 'ΑΕΣΙΤΝΟΡΙΛΚΜΔΠΗΦΧΘΓΒΖΩΞΨ'  # ~most→least frequent in Greek
nums_by_freq = [n for n, _ in sorted(num_freq.items(), key=lambda x: -x[1])]

# Map: most frequent numbers → most frequent letters
freqmap = {}
for i, n in enumerate(nums_by_freq):
    freqmap[n] = GREEK_FREQ[i % len(GREEK_FREQ)]

t0 = time.time()
freq_parts = [''.join(freqmap[n] for n in sorted(nums)) for _, nums in all_draws]
freq_stream = ''.join(freq_parts)
print(f"  Built freq_stream: {len(freq_stream):,} chars in {time.time()-t0:.1f}s")
print(f"  First 200: {freq_stream[:200]}")

# Search for words
print(f"\n  Word search (frequency-mapped):")
for word in ['ΚΑΙ','ΤΟ','ΤΗΝ','ΝΑΙ','ΕΙΝ','ΓΙΑ','ΑΠΟ','ΜΙΑ','ΕΝΑ','ΔΕΝ',
             'ΝΙΚΗ','ΚΙΝΟ','ΤΥΧΗ','ΕΛΛΑ','ΑΘΗΝΑ','ΕΛΛΑΣ']:
    count = freq_stream.count(word)
    if count > 0:
        print(f"    '{word}': {count:,}× ★")

# Top words in freq stream
freq_words = re.findall(r'[ΑΒΓΔΕΖΗΘΙΚΛΜΝΞΟΠΡΣΤΥΦΧΨΩ]{3,}', freq_stream)
freq_wc = Counter(freq_words).most_common(20)
print(f"\n  Top 20 words (3+ uppercase letters) in freq-mapped stream:")
for w, cnt in freq_wc:
    print(f"    '{w}': {cnt:,}×")

# ══════════════════════════════════════════════════════════════════
# Exp 6: Vowel/consonant pattern — does KINO produce Greek-like V/C patterns?
# ══════════════════════════════════════════════════════════════════
print("\n══ Experiment 6: Vowel/Consonant distribution ══")

VOWEL_NUMS  = {1,5,7,9,15,20,24}  # Α,Ε,Η,Ι,Ο,Υ,Ω
CONS_NUMS   = set(range(1,25)) - VOWEL_NUMS  # rest of uppercase
VOWEL_NUMS_lo = {25,29,31,33,39,44,48}  # α,ε,η,ι,ο,υ,ω
PUNCT_NUMS  = set(range(49,81))

for label, nums_set in [('Φωνήεντα (1-24)', VOWEL_NUMS),
                         ('Σύμφωνα (1-24)', CONS_NUMS),
                         ('Σημεία (49-80)', PUNCT_NUMS)]:
    total = sum(sum(1 for n in draw if n in nums_set) for _, draw in all_draws)
    per_draw = total / N
    expected = len(nums_set) * (20/80)
    print(f"  {label}: μ.ό.={per_draw:.3f}/κλήρωση  αναμένεται={expected:.3f}")

# Greek V/C ratio in natural text: ~45% vowels
# In KINO (uppercase section 1-24): 7 vowels / 24 = 29% vowels
# So KINO "text" has fewer vowels than natural Greek
vowel_rate = len(VOWEL_NUMS) / 24
print(f"\n  Ποσοστό φωνηέντων στο KINO uppercase: {vowel_rate*100:.1f}%")
print(f"  Ποσοστό φωνηέντων σε φυσικό ελληνικό κείμενο: ~45%")
print(f"  → KINO 'γλώσσα' έχει ΛΙΓΟΤΕΡΑ φωνήεντα από φυσική ομιλία")

# ══════════════════════════════════════════════════════════════════
# Exp 7: Perfect word draw — has any draw ever contained ΚΙΝΟ (10,9,13,15)?
# Detailed look at those draws
# ══════════════════════════════════════════════════════════════════
print("\n══ Experiment 7: Draws containing ΚΙΝΟ = {1,9,10,13,15} ══")
# Wait — Κ=10, Ι=9, Ν=13, Ο=15
KINO_SET = frozenset([10, 9, 13, 15])  # K,I,N,O
kino_draws = [(did, nums) for did, nums in all_draws if KINO_SET <= frozenset(nums)]
print(f"  Draws containing Κ(10)+Ι(9)+Ν(13)+Ο(15): {len(kino_draws):,}")
print(f"  Expected: {N * comb(76,16)/comb(80,20):.0f}")
if kino_draws:
    print(f"  Πρώτη: #{kino_draws[0][0]}  nums={sorted(kino_draws[0][1])}")
    print(f"  Τελευταία: #{kino_draws[-1][0]}  nums={sorted(kino_draws[-1][1])}")

# How often does shuffled version of such a draw spell ΚΙΝΟ consecutively?
if kino_draws:
    consecutive_kino = 0
    random.seed(99)
    for _, nums in kino_draws[:1000]:
        for _ in range(100):  # 100 shuffles
            s = nums[:]
            random.shuffle(s)
            text = ''.join(n2c(n) for n in s)
            if 'ΚΙΝΟ' in text:
                consecutive_kino += 1
                break
    print(f"  Στα 1000 πρώτα ΚΙΝΟ-draws, shuffled 100×: {consecutive_kino}/1000 φορές έγραψε 'ΚΙΝΟ' συνεχόμενα")

print("\n══ ΤΕΛΟΣ ΠΕΙΡΑΜΑΤΩΝ ══")
