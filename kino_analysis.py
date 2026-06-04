#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Full Kino Number-to-Letter Mapping Analysis
"""

import json
import os
import glob
from collections import Counter, defaultdict

# ─────────────────────────────────────────────
# 1. NUM_TO_CHAR mapping (as specified)
# ─────────────────────────────────────────────
NUM_TO_CHAR = {
    1: 'ευ', 2: 'τρ', 3: 'Ξ', 4: 'Χ', 5: 'η', 6: 'αυ', 7: 'χτ', 8: 'ψ', 9: 'λλ', 10: 'τ',
    11: 'μπ', 12: 'Υ', 13: 'γ', 14: 'φ', 15: 'οι', 16: 'κ', 17: 'Ω', 18: 'Ο', 19: 'Ψ', 20: 'σκ',
    21: 'Η', 22: 'ο', 23: 'αι', 24: 'Α', 25: 'Δ', 26: 'Ρ', 27: 'χ', 28: 'ρχ', 29: 'ντ', 30: 'Γ',
    31: 'ν', 32: 'θ', 33: 'σ', 34: 'δ', 35: 'χρ', 36: 'Σ', 37: 'Τ', 38: 'π', 39: 'ξ', 40: 'μ',
    41: 'υ', 42: 'Φ', 43: 'τσ', 44: 'λ', 45: 'ό', 46: 'Λ', 47: 'ή', 48: 'Μ', 49: 'θρ', 50: 'στ',
    51: 'ζ', 52: 'ου', 53: 'α', 54: 'Ζ', 55: 'ει', 56: 'σπ', 57: 'κτ', 58: 'νδ', 59: 'β', 60: 'Κ',
    61: 'γκ', 62: 'ύ', 63: 'ρ', 64: 'ω', 65: 'Ε', 66: 'σσ', 67: 'ί', 68: 'Β', 69: 'Ι', 70: 'φτ',
    71: 'Ν', 72: 'γγ', 73: 'ε', 74: 'ά', 75: 'Π', 76: 'πρ', 77: 'ώ', 78: 'Θ', 79: 'ι', 80: 'έ'
}

# ─────────────────────────────────────────────
# 2. Build lowercase multiset per draw
# ─────────────────────────────────────────────
# Normalize: lowercase all chars. This means Υ(12)→υ, Η(21)→η, Ο(18)→ο, etc.
# Each number contributes its character(s) lowercased.

def num_to_chars_lower(n):
    """Return list of lowercase chars for a number."""
    raw = NUM_TO_CHAR[n]
    return list(raw.lower())

# ─────────────────────────────────────────────
# 3. Build CHAR_TO_NUM (first occurrence wins, lowercase key)
# ─────────────────────────────────────────────
# We need patterns: multichar tokens first (longest-first for DFS matching)
CHAR_TO_NUM = {}
for num in range(1, 81):
    key = NUM_TO_CHAR[num].lower()
    if key not in CHAR_TO_NUM:
        CHAR_TO_NUM[key] = num

# All pattern strings sorted longest first, then alphabetically for determinism
PATTERNS = sorted(CHAR_TO_NUM.keys(), key=lambda x: (-len(x), x))

print("=== CHAR_TO_NUM (first-occurrence, lowercase) ===")
for k, v in sorted(CHAR_TO_NUM.items(), key=lambda x: x[1]):
    print(f"  {v:2d} → '{k}'")
print(f"\nTotal unique patterns: {len(PATTERNS)}")
print(f"Patterns sorted longest-first: {PATTERNS[:10]} ...")

# ─────────────────────────────────────────────
# 4. Load all draws
# ─────────────────────────────────────────────
DATA_DIR = '/home/user/Game/data/raw'
files = sorted(glob.glob(os.path.join(DATA_DIR, 'kino_raw_*.json')))

all_draws = []
months_loaded = []
for f in files:
    with open(f, 'r', encoding='utf-8') as fh:
        data = json.load(fh)
    draws = data.get('draws', [])
    all_draws.extend(draws)
    months_loaded.append(data.get('month', os.path.basename(f)))

print(f"\n=== DATA LOADED ===")
print(f"Files found: {len(files)}")
print(f"Months: {months_loaded[0]} → {months_loaded[-1]}")
print(f"Total draws: {len(all_draws)}")

# ─────────────────────────────────────────────
# 5. Build multiset for each draw (lowercase chars)
# ─────────────────────────────────────────────
def draw_multiset(draw_numbers):
    """Return Counter of lowercase char tokens (length-1 and length-2) for a draw."""
    c = Counter()
    for n in draw_numbers:
        raw = NUM_TO_CHAR[n].lower()
        # Store the full token (could be 1 or 2 chars) as atomic unit
        c[raw] += 1
    return c

draw_multisets = [draw_multiset(d['n']) for d in all_draws]

# ─────────────────────────────────────────────
# 6. DFS word-spelling checker
# ─────────────────────────────────────────────
def normalize_word(word):
    """Lowercase, replace ς→σ."""
    return word.lower().replace('ς', 'σ')

def can_spell(word, multiset):
    """
    Check if 'word' (already normalized) can be spelled from multiset
    using DFS with longest-first pattern matching.
    """
    norm = normalize_word(word)
    # DFS: pos in word, remaining multiset as Counter
    def dfs(pos, remaining):
        if pos == len(norm):
            return True
        for pat in PATTERNS:
            if norm.startswith(pat, pos):
                if remaining.get(pat, 0) > 0:
                    remaining[pat] -= 1
                    if remaining[pat] == 0:
                        del remaining[pat]
                    if dfs(pos + len(pat), remaining):
                        remaining[pat] = remaining.get(pat, 0) + 1
                        return True
                    remaining[pat] = remaining.get(pat, 0) + 1
        return False

    # Work on a copy
    m_copy = dict(multiset)
    return dfs(0, m_copy)

# ─────────────────────────────────────────────
# 7. Word lists
# ─────────────────────────────────────────────
cities = [
    'αθήνα', 'θεσσαλονίκη', 'πάτρα', 'λάρισα', 'ηράκλειο', 'βόλος', 'ρόδος',
    'κόρινθος', 'χανιά', 'κέρκυρα', 'ιωάννινα', 'καβάλα', 'τρίκαλα', 'λαμία',
    'κοζάνη', 'καλαμάτα', 'βέροια', 'σέρρες', 'μυτιλήνη', 'χίος', 'σάμος',
    'νάξος', 'μύκονος', 'σαντορίνη', 'ζάκυνθος', 'κρήτη', 'πύλος', 'σπάρτη',
    'δελφοί', 'ολυμπία', 'μεσσήνη', 'τίρυνθα', 'κνωσός', 'ελευσίνα', 'μαραθώνας',
    'σαλαμίνα', 'θερμοπύλες', 'μετέωρα'
]

names = [
    'νίκος', 'μαρία', 'γιώργος', 'ελένη', 'κώστας', 'δήμητρα', 'αντώνης',
    'σοφία', 'νικόλαος', 'ιωάννης', 'χρήστος', 'παναγιώτης', 'θανάσης',
    'στέφανος', 'λευτέρης', 'σπύρος', 'σταύρος', 'ανδρέας', 'αλέξανδρος',
    'θεόδωρος', 'χρυσή', 'μελίνα', 'αλεξάνδρα', 'θεοδώρα', 'ευαγγελία',
    'χριστίνα', 'σταματία', 'βασίλης', 'δημήτρης', 'κατερίνα', 'αθανασία',
    'παρασκευή', 'ευθυμία', 'αγγελική', 'ευαγγελίνα'
]

emotions = [
    'αγάπη', 'φόβος', 'χαρά', 'λύπη', 'οργή', 'ελπίδα', 'πόνος', 'ειρήνη',
    'δύναμη', 'πάθος', 'θυμός', 'ζήλεια', 'ντροπή', 'ευτυχία', 'δυστυχία',
    'αγωνία', 'έκπληξη', 'ηρεμία', 'γαλήνη', 'αγανάκτηση', 'απελπισία',
    'ανακούφιση', 'ανησυχία', 'σύγχυση', 'τρόμος', 'δέος', 'θαύμα', 'έρωτας',
    'μίσος', 'απογοήτευση', 'περηφάνεια', 'ντροπή', 'χαρούμενος', 'θλίψη'
]

other = [
    'νερό', 'φωτιά', 'γη', 'ουρανός', 'θάλασσα', 'ήλιος', 'σελήνη', 'αστέρι',
    'ανεμος', 'βουνό', 'δέντρο', 'λουλούδι', 'ζωή', 'θάνατος', 'αρχή', 'τέλος',
    'αλήθεια', 'ψέμα', 'δικαιοσύνη', 'ελευθερία', 'ευτυχία', 'μνήμη', 'όνειρο',
    'πραγματικότητα', 'καλοκαίρι', 'χειμώνας', 'άνοιξη', 'φθινόπωρο', 'αυγή',
    'νύχτα', 'μέρα'
]

word_lists = {
    'cities': cities,
    'names': names,
    'emotions': emotions,
    'other': other,
}

# Deduplicate all words
all_words = list(dict.fromkeys(
    w for wlist in word_lists.values() for w in wlist
))
print(f"\nTotal unique words to check: {len(all_words)}")

# ─────────────────────────────────────────────
# 8. Count how many draws can spell each word
# ─────────────────────────────────────────────
print("\nRunning DFS check across all draws and words (this may take a minute)...")

word_counts = {}
N = len(all_draws)

for idx, word in enumerate(all_words):
    norm = normalize_word(word)
    count = 0
    for ms in draw_multisets:
        if can_spell(norm, ms):
            count += 1
    word_counts[word] = count
    if (idx + 1) % 10 == 0:
        print(f"  Checked {idx+1}/{len(all_words)} words...")

print(f"Done! Checked {len(all_words)} words against {N} draws.")

# ─────────────────────────────────────────────
# 9. Results
# ─────────────────────────────────────────────
print("\n" + "="*70)
print("STEP 6 — RESULTS")
print("="*70)

# Sort all words by count descending
sorted_words = sorted(word_counts.items(), key=lambda x: -x[1])

print(f"\n--- TOP 20 MOST FREQUENTLY FORMABLE WORDS (out of {N} draws) ---")
for rank, (word, cnt) in enumerate(sorted_words[:20], 1):
    pct = cnt / N * 100
    print(f"  {rank:2d}. {word:<25s}  {cnt:5d} / {N}  ({pct:.2f}%)")

print("\n--- TOP 10 PER CATEGORY ---")
for cat, wlist in word_lists.items():
    cat_sorted = sorted([(w, word_counts[w]) for w in wlist], key=lambda x: -x[1])
    print(f"\n  [{cat.upper()}]")
    for rank, (word, cnt) in enumerate(cat_sorted[:10], 1):
        pct = cnt / N * 100
        print(f"    {rank:2d}. {word:<25s}  {cnt:5d} ({pct:.2f}%)")

# ─────────────────────────────────────────────
# 10. Letter needs for top 50 words
# ─────────────────────────────────────────────
print("\n--- LETTER/TILE NEEDS FOR TOP 50 WORDS ---")
top50 = [w for w, _ in sorted_words[:50]]
tile_need = Counter()
for word in top50:
    norm = normalize_word(word)
    # Count which patterns are needed (greedy left-to-right for analysis)
    pos = 0
    while pos < len(norm):
        matched = False
        for pat in PATTERNS:
            if norm.startswith(pat, pos):
                tile_need[pat] += 1
                pos += len(pat)
                matched = True
                break
        if not matched:
            pos += 1  # skip unmatched char (shouldn't happen)
tile_need_sorted = tile_need.most_common(30)
print("  (Greedy left-to-right decomposition of top-50 words)")
for pat, cnt in tile_need_sorted:
    num = CHAR_TO_NUM.get(pat, '?')
    print(f"  pattern '{pat}' (num {num}): needed {cnt} times across top-50 words")

# ─────────────────────────────────────────────
# 11. Letter frequency across ALL draws
# ─────────────────────────────────────────────
print("\n--- CHAR/PATTERN FREQUENCY ACROSS ALL DRAWS ---")
global_char_freq = Counter()
for ms in draw_multisets:
    for tok, cnt in ms.items():
        global_char_freq[tok] += cnt

print("  (Each draw has 20 numbers, total tokens = 20 × N)")
print(f"  Total token occurrences: {sum(global_char_freq.values())}")
total_tokens = sum(global_char_freq.values())
for tok, cnt in global_char_freq.most_common(40):
    num = CHAR_TO_NUM.get(tok, '?')
    pct = cnt / total_tokens * 100
    # Map back to original numbers that produce this token
    orig_nums = [n for n in range(1, 81) if NUM_TO_CHAR[n].lower() == tok]
    print(f"  '{tok}' (nums {orig_nums}): {cnt} ({pct:.3f}%)")

# ─────────────────────────────────────────────
# 12. Frequency of each number (how often it appears across all draws)
# ─────────────────────────────────────────────
print("\n--- NUMBER FREQUENCY ACROSS ALL DRAWS ---")
num_freq = Counter()
for d in all_draws:
    for n in d['n']:
        num_freq[n] += 1
total_num = sum(num_freq.values())
for num in sorted(num_freq.keys()):
    cnt = num_freq[num]
    pct = cnt / total_num * 100
    print(f"  num {num:2d} ('{NUM_TO_CHAR[num]}'): {cnt} ({pct:.3f}%)")

# ─────────────────────────────────────────────
# 13. STEP 7 — Proposed new mapping
# ─────────────────────────────────────────────
print("\n" + "="*70)
print("STEP 7 — PROPOSED NEW MAPPING (24 Greek letters, 80 numbers)")
print("="*70)

# 24 Greek letters with rough frequency in Modern Greek text
# We'll compute desired distribution from target word letter frequencies

# Collect all letters needed by all words (normalized)
word_letter_freq = Counter()
for word in all_words:
    norm = normalize_word(word)
    for ch in norm:
        # strip accents for base letter
        import unicodedata
        base = unicodedata.normalize('NFD', ch)
        base = ''.join(c for c in base if unicodedata.category(c) != 'Mn')
        if base:
            word_letter_freq[base] += 1

print("\n  Letter frequency in all target words (base letters, unaccented):")
total_letters = sum(word_letter_freq.values())
letter_pct = {}
for ch, cnt in word_letter_freq.most_common():
    pct = cnt / total_letters * 100
    letter_pct[ch] = pct
    print(f"  '{ch}': {cnt} ({pct:.2f}%)")

# The 24 standard Greek letters
greek_24 = ['α', 'β', 'γ', 'δ', 'ε', 'ζ', 'η', 'θ', 'ι', 'κ', 'λ', 'μ',
            'ν', 'ξ', 'ο', 'π', 'ρ', 'σ', 'τ', 'υ', 'φ', 'χ', 'ψ', 'ω']

# Each number independently uniform (p=1/80 each), so ideal: assign numbers
# proportional to letter frequency. Each letter gets floor(80 * pct/100) numbers
# minimum 1, then distribute remainder.
# Filter to only the 24 letters
filtered = {ch: letter_pct.get(ch, 0) for ch in greek_24}
# Redistribute: normalize to sum=1
total_f = sum(filtered.values())
if total_f == 0:
    total_f = 1
norm_f = {ch: v / total_f for ch, v in filtered.items()}

# Raw allocation using floor + largest-remainder method
raw_alloc = {ch: max(1, int(80 * norm_f[ch])) for ch in greek_24}
curr_sum = sum(raw_alloc.values())
remainder = 80 - curr_sum
# Sort by fractional part descending
fracs = sorted(greek_24, key=lambda c: -(80 * norm_f[c] - int(80 * norm_f[c])))
for i in range(remainder):
    raw_alloc[fracs[i]] += 1

assert sum(raw_alloc.values()) == 80, f"Sum={sum(raw_alloc.values())}"

print("\n  Proposed number allocation per letter (proportional to word-letter frequency):")
for ch in sorted(raw_alloc.keys()):
    print(f"  '{ch}': {raw_alloc[ch]} numbers")

# Now assign actual numbers to letters based on number frequency:
# Assign numbers that appear MOST in draws to letters that MOST need coverage
# Sort numbers by draw frequency (most frequent first)
sorted_nums_by_freq = [n for n, _ in num_freq.most_common()]
# Sort letters by desired allocation descending
sorted_letters_by_alloc = sorted(raw_alloc.keys(), key=lambda c: -raw_alloc[c])

new_mapping = {}
num_idx = 0
for letter in sorted_letters_by_alloc:
    alloc = raw_alloc[letter]
    for _ in range(alloc):
        new_mapping[sorted_nums_by_freq[num_idx]] = letter
        num_idx += 1

print("\n  NEW NUM → LETTER mapping (numbers sorted by draw frequency, letters by alloc):")
print("  Format: num → letter (draw_freq)")
for num in range(1, 81):
    letter = new_mapping[num]
    freq = num_freq[num]
    print(f"  {num:2d} → {letter}  (freq={freq})")

print("\n  Summary — numbers per letter in new mapping:")
letter_num_map = defaultdict(list)
for num, letter in new_mapping.items():
    letter_num_map[letter].append(num)
for letter in greek_24:
    nums = sorted(letter_num_map[letter])
    print(f"  '{letter}': {len(nums)} numbers → {nums}")

print("\n=== ANALYSIS COMPLETE ===")
