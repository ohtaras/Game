#!/usr/bin/env python3
"""
KINO as text: map 80 numbers to 80 characters (Greek letters + punctuation)
Each draw of 20 numbers → 20 characters → a fragment of "story"
All draws in a day → a full "story"

Mapping:
  1-24  → Α Β Γ Δ Ε Ζ Η Θ Ι Κ Λ Μ Ν Ξ Ο Π Ρ Σ Τ Υ Φ Χ Ψ Ω  (Greek uppercase)
  25-48 → α β γ δ ε ζ η θ ι κ λ μ ν ξ ο π ρ σ τ υ φ χ ψ ω  (Greek lowercase)
  49    → ' ' (space)
  50    → '.'   51 → ','   52 → '!'   53 → '?'   54 → ';'
  55    → ':'   56 → '-'   57 → '('   58 → ')'   59 → '"'
  60    → '«'   61 → '»'   62 → '\n'  63 → '…'   64 → '/'
  65-74 → 0-9 (digits)
  75-80 → @ # & * + =
"""
import json, time, random, re
from pathlib import Path
from collections import Counter

DATA_DIR = Path('/home/user/Game/data/raw')

# ── Character mapping ────────────────────────────────────────────────────
GR_UP  = 'ΑΒΓΔΕΖΗΘΙΚΛΜΝΞΟΠΡΣΤΥΦΧΨΩ'   # 24 chars, 1-24
GR_LOW = 'αβγδεζηθικλμνξοπρστυφχψω'   # 24 chars, 25-48
PUNCT  = ' .,!?;:-()\"«»\n…/0123456789@#&*+='  # 32 chars, 49-80

CHARMAP = [None]  # 1-indexed
for c in GR_UP:   CHARMAP.append(c)   # 1-24
for c in GR_LOW:  CHARMAP.append(c)   # 25-48
for c in PUNCT:   CHARMAP.append(c)   # 49-80

assert len(CHARMAP) == 81, f"Expected 81, got {len(CHARMAP)}"

def nums_to_text(nums, sort=True):
    ns = sorted(nums) if sort else list(nums)
    return ''.join(CHARMAP[n] for n in ns if CHARMAP[n] is not None)

# ── Load draws ───────────────────────────────────────────────────────────
print("Loading draws...")
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

# ── Show character mapping ───────────────────────────────────────────────
print("\n── Χαρακτήρας ανά αριθμό ──")
for n in range(1, 81):
    c = CHARMAP[n]
    print(f"  {n:2d}→'{c}'", end='\n' if n%10==0 else '  ')

# ── Generate text for a few sample days ─────────────────────────────────
# Estimate: ~272 draws per day, anchor draw 1303293 = 2026-05-31
ANCHOR_ID   = 1303293
DRAWS_PER_DAY = 272

def day_text(start_idx, n_draws=272, separator='|', sort_nums=True):
    """Concatenate text from n_draws consecutive draws."""
    fragments = []
    for i in range(start_idx, min(start_idx + n_draws, N)):
        nums = all_draws[i][1]
        t = nums_to_text(nums, sort=sort_nums)
        fragments.append(t)
    return separator.join(fragments)

print("\n\n── Δείγμα: 5 πρώτες κληρώσεις (ταξινομημένοι αριθμοί) ──")
for i in range(5):
    draw_id, nums = all_draws[i]
    txt = nums_to_text(nums, sort=True)
    print(f"  Draw #{draw_id}: nums={sorted(nums)}")
    print(f"           text= {txt}")

print("\n── Δείγμα: πρώτη 'μέρα' (272 κληρώσεις) χωρίς διαχωριστικό ──")
day0 = ''.join(nums_to_text(all_draws[i][1]) for i in range(272))
print(f"  Μήκος: {len(day0)} χαρακτήρες")
print(f"  Πρώτοι 500:\n{day0[:500]}")

# ── Without sort: draw order ─────────────────────────────────────────────
print("\n── Δείγμα: 5 πρώτες κληρώσεις (ΜΗ ταξινομημένοι) ──")
for i in range(5):
    draw_id, nums = all_draws[i]
    txt = nums_to_text(nums, sort=False)
    print(f"  Draw #{draw_id}: {txt}")

# ── Word search: find real Greek words in the stream ────────────────────
print("\n── Αναζήτηση ελληνικών λέξεων στο 'κείμενο' ──")

# Generate full text stream (sorted, space-separated fragments as "words")
# Each draw is a "word" of 20 chars — search within each draw fragment
GREEK_WORDS = [
    'ΚΑΙ','ΤΟ','ΤΗ','ΤΑ','ΟΙ','ΜΕ','ΜΙΑ','ΕΝΑ','ΔΕΝ','ΝΑΙ',
    'ΚΙΝΟ','ΤΥΧΗ','ΝΙΚΗ','ΑΡΙΘΜΟΣ','ΚΛΗΡΩΣΗ',
    'αι','εν','ον','ος','ης','ων','κι','να','με','σε','τι',
    'ΝΙΚΩ','ΦΩΣ','ΓΗ','ΘΕΩ','ΩΔΗ',
]

full_stream = ''.join(nums_to_text(d[1]) for d in all_draws)
print(f"  Συνολικό κείμενο: {len(full_stream):,} χαρακτήρες")

for word in GREEK_WORDS:
    count = full_stream.count(word)
    # Expected: (20/80)^len(word) per position × N×20 positions
    p_per_char = (20/80) ** len(word)
    expected = len(full_stream) * p_per_char
    ratio = count/expected if expected > 0 else 0
    print(f"  '{word}' ({len(word)}γρ): βρέθηκε {count:6,}×  αναμενόμενο {expected:7.0f}×  ratio={ratio:.3f}")

# ── Most common 3-letter sequences ──────────────────────────────────────
print("\n── Top 20 τριγράμματα (3-char sequences) ──")
trigrams = Counter(full_stream[i:i+3] for i in range(len(full_stream)-2))
for tri, cnt in trigrams.most_common(20):
    print(f"  '{tri}': {cnt:,}")

# ── Most common "words" (runs of letters between spaces/punctuation) ──
print("\n── Top 20 'λέξεις' (συνεχόμενα γράμματα) ──")
words_found = re.findall(r'[ΑΒΓΔΕΖΗΘΙΚΛΜΝΞΟΠΡΣΤΥΦΧΨΩαβγδεζηθικλμνξοπρστυφχψω]+', full_stream)
word_counter = Counter(w for w in words_found if len(w) >= 3)
for w, cnt in word_counter.most_common(20):
    print(f"  '{w}': {cnt:,}×")

# ── Artistic output: format one "day" as prose ──────────────────────────
print("\n\n════════════════════════════════════════════════════════")
print("ΚΙΝΟ ΙΣΤΟΡΙΑ — Μέρα 1 (κληρώσεις 1–272):")
print("════════════════════════════════════════════════════════")
# Format: every 10 draws = one paragraph, numbers sorted
prose = ''
for i in range(min(272, N)):
    txt = nums_to_text(all_draws[i][1], sort=True)
    prose += txt
    if (i+1) % 10 == 0:
        prose += '\n'
    else:
        prose += ' '
print(prose[:2000])

print("\n── Ίδια μέρα χωρίς ταξινόμηση (draw order): ──")
prose2 = ''
for i in range(min(272, N)):
    txt = nums_to_text(all_draws[i][1], sort=False)
    prose2 += txt + ' '
    if (i+1) % 10 == 0:
        prose2 = prose2.rstrip() + '\n'
print(prose2[:1000])
