#!/usr/bin/env python3
"""
Batch 12: Φράσεις & Λεξαρίθμοι — Διορθωμένη & Επεκταμένη Ανάλυση
Πλήρως numpy-vectorized για ταχύτητα.

Tests:
  T1: Φράση-ως-υποσύνολο (θέσης 1-24)
  T2: Φράση-ως-υποσύνολο (Μιλήσιο 1-80)
  T3: Μερικό match ≥K/N
  T4: Διορθ. Εκπλήρωση (EXACTLY k → υπόλοιπα επόμενη)
  T5: Ένωση 2 συνεχόμενων κληρώσεων
  T6: Εβδομαδιαία bias
  T7: Ψηφιακή ρίζα αθροίσματος → αριθμός
  T8: Cross-draw resonance (lag 1-17)
  T9: Back-projection (φράση@i → αριθμοί@i-1)
  T10: Ισοψηφία draw_sum = Μιλήσιος
"""

import json, glob, re
from collections import defaultdict, Counter
from math import comb
import numpy as np
from datetime import datetime, timedelta

# ═══════════════════════════════════════════════════════════════════
# Δεδομένα
# ═══════════════════════════════════════════════════════════════════
draws_raw = []
for f in sorted(glob.glob('/home/user/Game/data/raw/kino_raw_*.json')):
    with open(f) as fp:
        d = json.load(fp)
    draws_raw.extend(d['draws'])
draws_raw.sort(key=lambda x: x['id'])
N = len(draws_raw)
M = [set(d['n']) for d in draws_raw]
IDS = np.array([d['id'] for d in draws_raw])
print(f"Κληρώσεις: {N:,}")

mat = np.zeros((N, 81), dtype=np.int8)
for i, d in enumerate(draws_raw):
    for n in d['n']:
        mat[i, n] = 1

draw_sums = mat[:, 1:].sum(axis=1) * 0  # placeholder
draw_sums = np.array([sum(d['n']) for d in draws_raw], dtype=np.int32)
draw_dr = np.array([1 + (int(s)-1) % 9 for s in draw_sums], dtype=np.int8)

# Εβδομαδιαία ημέρα από ID
BASE_ID = 1062550
BASE_DATE = datetime(2024, 1, 2)
DRAWS_PER_DAY = 272.0
days_offset = (IDS - BASE_ID) / DRAWS_PER_DAY
draw_weekdays = np.array([(BASE_DATE + timedelta(days=float(d))).weekday()
                           for d in days_offset], dtype=np.int8)
WEEKDAY_NAMES = ['Δευ','Τρι','Τετ','Πεμ','Παρ','Σαβ','Κυρ']

# ═══════════════════════════════════════════════════════════════════
# Λεξαρίθμοι
# ═══════════════════════════════════════════════════════════════════
MILESIAN = {
    'Α':1,'Β':2,'Γ':3,'Δ':4,'Ε':5,'Ζ':7,'Η':8,'Θ':9,
    'Ι':10,'Κ':20,'Λ':30,'Μ':40,'Ν':50,'Ξ':60,'Ο':70,'Π':80,
    'Ρ':100,'Σ':200,'Τ':300,'Υ':400,'Φ':500,'Χ':600,'Ψ':700,'Ω':800,
}
MILESIAN_1_80 = {c:v for c,v in MILESIAN.items() if v <= 80}
GR_ORDER = 'ΑΒΓΔΕΖΗΘΙΚΛΜΝΞΟΠΡΣΤΥΦΧΨΩ'
POSITIONAL = {c: i+1 for i, c in enumerate(GR_ORDER)}

def normalize(text):
    text = text.upper()
    for a, b in {'Ά':'Α','Έ':'Ε','Ή':'Η','Ί':'Ι','Ό':'Ο','Ύ':'Υ','Ώ':'Ω',
                 'Ϊ':'Ι','Ϋ':'Υ','ΐ':'Ι','ΰ':'Υ','Ϲ':'Σ'}.items():
        text = text.replace(a, b)
    return re.sub(r'[^ΑΒΓΔΕΖΗΘΙΚΛΜΝΞΟΠΡΣΤΥΦΧΨΩ]', '', text)

def milesian(text):
    return sum(MILESIAN.get(c, 0) for c in normalize(text))

def positional_nums(text):
    return [POSITIONAL[c] for c in normalize(text) if c in POSITIONAL]

def milesian_nums_1_80(text):
    return [MILESIAN_1_80[c] for c in normalize(text) if c in MILESIAN_1_80]

def digital_root(n):
    return 0 if n == 0 else 1 + (n - 1) % 9

# ═══════════════════════════════════════════════════════════════════
# Φράσεις
# ═══════════════════════════════════════════════════════════════════
PHRASES = [
    ("ΚΙΝΟ","παιχνίδι"),("ΤΥΧΗ","τύχη"),("ΚΑΛΗ ΤΥΧΗ","ευχή"),
    ("ΚΕΡΔΟΣ","κέρδος"),("ΝΙΚΗ","νίκη"),("ΧΡΗΜΑ","χρήμα"),
    ("ΠΛΟΥΤΟΣ","πλούτος"),("ΤΖΑΚΠΟΤ","jackpot"),("ΔΕΛΤΙΟ","δελτίο"),
    ("ΑΡΙΘΜΟΣ","αριθμός"),("ΚΛΗΡΩΣΗ","κλήρωση"),("ΚΛΗΡΟΣ","κλήρος"),
    ("ΕΚΤΑΚΤΗ ΤΥΧΗ","έκτ.τύχη"),("ΜΕΓΑΛΟ ΚΕΡΔΟΣ","μεγ.κέρδος"),
    ("ΤΥΧΑΙΑ ΕΠΙΛΟΓΗ","random"),("ΤΑ ΝΟΥΜΕΡΑ","αριθμοί"),
    ("ΝΙΚΗΤΗΣ","νικητής"),("ΤΟ ΤΖΑΚΠΟΤ","jackpot"),
    ("ΠΡΩΤΟΣ ΑΡΙΘΜΟΣ","prime"),("ΤΥΧΑΙΟΣ ΑΡΙΘΜΟΣ","random"),
    ("ΘΕΟΣ","Θεός"),("ΚΥΡΙΕ ΕΛΕΗΣΟΝ","Κύρ.ελ."),
    ("ΚΥΡΙΟΣ","Κύριος"),("ΙΗΣΟΥΣ ΧΡΙΣΤΟΣ","ΙΧ"),
    ("ΙΗΣΟΥΣ","Ιησούς"),("ΧΡΙΣΤΟΣ","Χριστός"),
    ("ΧΡΙΣΤΟΣ ΑΝΕΣΤΗ","ΧΑ"),("ΑΛΗΘΩΣ ΑΝΕΣΤΗ","ΑΑ"),
    ("ΠΑΝΑΓΙΑ","Παναγία"),("ΘΕΟΤΟΚΟΣ","Θεοτόκος"),
    ("ΜΑΡΙΑ","Μαρία"),("ΑΓΙΟΣ","Άγιος"),
    ("ΑΓΙΑ ΤΡΙΑΣ","Αγ.Τρ."),
    ("ΕΝ ΑΡΧΗ ΗΝ Ο ΛΟΓΟΣ","Ιωάν.1:1"),
    ("ΛΟΓΟΣ","λόγος"),("ΠΝΕΥΜΑ ΑΓΙΟΝ","Πνεύμα"),
    ("ΒΑΣΙΛΕΙΑ ΤΩΝ ΟΥΡΑΝΩΝ","βασιλεία"),
    ("ΑΓΑΠΗ","αγάπη"),("ΠΙΣΤΙΣ","πίστη"),
    ("ΕΛΠΙΣ","ελπίδα"),("ΣΟΦΙΑ","σοφία"),
    ("ΑΛΗΘΕΙΑ","αλήθεια"),("ΖΩΗ","ζωή"),
    ("ΦΩΣ","φως"),("ΣΚΟΤΟΣ","σκότος"),
    ("ΑΓΓΕΛΟΣ","άγγελος"),("ΟΥΡΑΝΟΣ","ουρανός"),
    ("ΠΑΡΑΔΕΙΣΟΣ","παράδεισος"),("ΑΜΗΝ","αμήν"),
    ("ΑΛΛΗΛΟΥΙΑ","αλληλούια"),("ΔΟΞΑ","δόξα"),
    ("ΔΟΞΑ ΤΩ ΘΕΩ","δόξα"),("ΑΝΑΣΤΑΣΗ","ανάσταση"),
    ("ΒΑΠΤΙΣΜΑ","βάπτισμα"),("ΜΕΤΑΝΟΙΑ","μετάνοια"),
    ("ΘΕΙΑ ΧΑΡΙΣ","θεία χάρις"),("ΕΚΚΛΗΣΙΑ","εκκλησία"),
    ("ΟΡΘΟΔΟΞΙΑ","ορθοδοξία"),
    ("ΑΓΙΟΣ ΓΕΩΡΓΙΟΣ","Άγ.Γεώρ."),("ΑΓΙΟΣ ΝΙΚΟΛΑΟΣ","Άγ.Νικ."),
    ("ΑΓΙΟΣ ΔΗΜΗΤΡΙΟΣ","Άγ.Δημ."),
    ("ΑΡΧΑΓΓΕΛΟΣ ΜΙΧΑΗΛ","Αρχ.Μιχ."),
    ("ΑΡΧΑΓΓΕΛΟΣ ΓΑΒΡΙΗΛ","Αρχ.Γαβ."),
    ("ΓΝΩΘΙ ΣΑΥΤΟΝ","Δελφοί"),("ΜΗΔΕΝ ΑΓΑΝ","Δελφοί"),
    ("ΕΝ ΟΙΔΑ ΟΤΙ ΟΥΔΕΝ ΟΙΔΑ","Σωκρ."),
    ("ΠΑΝΤΑ ΡΕΙ","Ηράκλ."),("ΛΟΓΟΣ ΤΟΥ ΠΑΝΤΟΣ","λόγ.παντ."),
    ("ΤΟ ΕΝ ΚΑΙ ΤΟ ΠΑΝ","Ηράκλ."),
    ("ΑΡΙΘΜΟΣ ΕΣΤΙΝ Η ΑΡΧΗ","Πυθαγ."),
    ("ΖΕΥΣ","Ζεύς"),("ΑΠΟΛΛΩΝ","Απόλλων"),
    ("ΕΡΜΗΣ","Ερμής"),("ΑΡΤΕΜΙΣ","Άρτεμης"),
    ("ΑΦΡΟΔΙΤΗ","Αφροδίτη"),("ΤΥΧΗ ΑΓΑΘΗ","τ.αγαθή"),
    ("ΚΑΛΟΣ ΚΑΓΑΘΟΣ","καλ."),("ΦΙΛΟΣΟΦΙΑ","φιλοσ."),
    ("ΨΥΧΗ ΤΟΥ ΚΟΣΜΟΥ","world soul"),
    ("ΣΩΦΡΟΣΥΝΗ","σωφρ."),("ΔΙΚΑΙΟΣΥΝΗ","δικαιοσ."),
    ("ΑΝΔΡΕΙΑ","ανδρεία"),("ΦΡΟΝΗΣΙΣ","φρόνηση"),
    ("ΕΛΛΑΣ","Ελλάς"),("ΕΛΛΗΝ","Έλλην"),
    ("ΕΛΕΥΘΕΡΙΑ","ελευθ."),("ΔΗΜΟΚΡΑΤΙΑ","δημοκρ."),
    ("ΠΑΤΡΙΔΑ","πατρίδα"),
    ("ΕΠΤΑ","7"),("ΕΝΝΕΑ","9"),("ΔΩΔΕΚΑ","12"),
    ("ΕΙΚΟΣΙ","20"),("ΤΕΣΣΑΡΑΚΟΝΤΑ","40"),
    ("ΑΛΦΑ ΚΑΙ ΩΜΕΓΑ","Α+Ω"),
    ("ΑΡΓΥΡΟΠΟΥΛΟΣ","Αργ/λος"),("ΚΟΣΜΟΣ","κόσμος"),
    ("ΕΙΜΑΡΜΕΝΗ","ειμαρμ."),("ΜΟΙΡΑ","μοίρα"),
    ("ΑΝΑΓΚΗ","ανάγκη"),("ΠΕΠΡΩΜΕΝΟ","πεπρ."),
    ("ΧΡΟΝΟΣ","χρόνος"),("ΑΙΩΝ","αιών"),
    ("ΜΟΝΑΔΑ","μονάδα"),("ΤΡΙΑΔΑ","τριάδα"),
    ("ΚΥΚΛΟΣ","κύκλος"),("ΤΕΤΡΑΓΩΝΟ","τετράγ."),
    ("ΤΡΙΓΩΝΟ","τρίγωνο"),("ΣΦΑΙΡΑ","σφαίρα"),
    ("ΧΡΥΣΟΣ","χρυσός"),("ΑΡΓΥΡΟΣ","άργυρος"),
    ("ΑΔΑΜ","Adam"),("ΑΒΡΑΑΜ","Αβρ."),
    ("ΜΩΥΣΗΣ","Μωυσ."),("ΔΑΒΙΔ","Δαβίδ"),
    ("ΣΟΛΟΜΩΝ","Σολομ."),("ΙΩΑΝΝΗΣ","Ιωάν."),
    ("ΠΑΥΛΟΣ","Παύλος"),("ΠΕΤΡΟΣ","Πέτρος"),
    ("ΓΕΩΡΓΙΟΣ","Γεώρ."),("ΝΙΚΟΛΑΟΣ","Νικόλ."),
    ("ΔΗΜΗΤΡΙΟΣ","Δημήτρ."),("ΚΩΝΣΤΑΝΤΙΝΟΣ","Κων/νος"),
    ("ΕΛΕΝΗ","Ελένη"),("ΝΙΚΟΣ","Νίκος"),
    ("ΚΩΣΤΑΣ","Κώστας"),("ΜΑΡΙΑ","Μαρία"),
    ("ΑΝΝΑ","Άννα"),
    ("ΤΟ ΦΩΣ ΤΟΥ ΚΟΣΜΟΥ","Ιωάν 8:12"),
    ("ΕΓΩ ΕΙΜΙ Η ΟΔΟ","Ιωάν 14:6"),
    ("ΑΓΑΠΑΤΕ ΑΛΛΗΛΟΥΣ","Ιωάν 13:34"),
    ("ΟΙ ΕΣΧΑΤΟΙ ΕΣΟΝΤΑΙ ΠΡΩΤΟΙ","Ματθ"),
    ("ΒΑΣΙΛΕΙΑ ΤΟΥ ΘΕΟΥ","βασιλ."),
    ("ΨΥΧΗ","ψυχή"),("ΣΩΜΑ","σώμα"),("ΝΟΥΣ","νους"),
    ("ΚΑΡΔΙΑ","καρδιά"),("ΝΟΥΣ ΚΑΙ ΚΑΡΔΙΑ","ν+κ"),
    ("ΛΟΓΟΣ ΘΕΟΥ","λόγ.Θ."),("ΤΟ ΑΓΙΟΝ ΠΝΕΥΜΑ","ΤΑΠ"),
    ("ΗΛΙΟΣ","ήλιος"),("ΣΕΛΗΝΗ","σελήνη"),
    ("ΑΣΤΡΑ","άστρα"),("ΠΑΝΣΕΛΗΝΟΣ","πανσ."),
    ("ΝΕΑ ΣΕΛΗΝΗ","νέα σελ."),
    ("ΚΥΡΙΑΚΗ","Κυριακή"),("ΣΑΒΒΑΤΟ","Σάββατο"),
    ("ΔΕΥΤΕΡΑ","Δευτέρα"),("ΤΡΙΤΗ","Τρίτη"),
    ("ΤΕΤΑΡΤΗ","Τετάρτη"),("ΠΕΜΠΤΗ","Πέμπτη"),
    ("ΠΑΡΑΣΚΕΥΗ","Παρ/κευή"),
    ("ΙΑΝΟΥΑΡΙΟΣ","Ιαν"),("ΦΕΒΡΟΥΑΡΙΟΣ","Φεβ"),
    ("ΜΑΡΤΙΟΣ","Μαρ"),("ΑΠΡΙΛΙΟΣ","Απρ"),
    ("ΜΑΙΟΣ","Μάι"),("ΙΟΥΝΙΟΣ","Ιουν"),
    ("ΙΟΥΛΙΟΣ","Ιουλ"),("ΑΥΓΟΥΣΤΟΣ","Αυγ"),
    ("ΣΕΠΤΕΜΒΡΙΟΣ","Σεπ"),("ΟΚΤΩΒΡΙΟΣ","Οκτ"),
    ("ΝΟΕΜΒΡΙΟΣ","Νοε"),("ΔΕΚΕΜΒΡΙΟΣ","Δεκ"),
    ("ΤΥΧΗ ΚΑΙ ΑΝΑΓΚΗ","τ+α"),
    ("ΑΓΑΠΗ ΚΑΙ ΣΟΦΙΑ","αγ+σ"),
    ("ΦΩΣ ΚΑΙ ΑΛΗΘΕΙΑ","φ+α"),
    ("ΘΕΟΣ ΑΓΑΠΗ ΕΣΤΙΝ","ΘΑΕ"),
    ("ΕΝ ΑΡΧ ΗΝ Ο ΛΟΓΟΣ","Ιωάν"),
    ("Ο ΧΡΙΣΤΟΣ ΑΝΕΣΤΗ","ΟΧΑ"),
    ("ΑΡΧΗ ΚΑΙ ΤΕΛΟΣ","α+τ"),
    ("ΙΩΑΝΝΗΣ Ο ΘΕΟΛΟΓΟΣ","ΙΘ"),
    ("ΠΑΥΛΟΣ Ο ΑΠΟΣΤΟΛΟΣ","ΠΑ"),
    ("ΑΓΙΑ ΣΟΦΙΑ","ΑΣ"),
    ("Η ΒΙΒΛΟΣ","Βίβλος"),("Η ΑΠΟΚΑΛΥΨΗ","Αποκ."),
    ("ΑΓΙΟΣ ΤΟΠΟΣ","Άγ.Τόπ."),
    ("ΣΗΜΕΡΑ ΝΙΚΗΣΑ","σημ.νίκ."),
    ("ΑΥΡΙΟ ΝΙΚΩ","αύρ.νικ."),
    ("ΕΝ ΑΡΧΗ ΗΝ Ο ΛΟΓΟΣ ΚΑΙ Ο ΛΟΓΟΣ ΗΝ ΠΡΟΣ ΤΟΝ ΘΕΟΝ","Ιωάν1:1full"),
    ("ΧΡΙΣΤΕ ΕΛΕΗΣΟΝ","Χρ.ελ."),
    ("ΖΩΟΔΟΧΟΣ ΠΗΓΗ","Ζωοδ.Πηγή"),
    ("ΘΕΟΤΟΚΕ ΒΟΗΘΕΙ","Θεοτ.Βοηθ"),
    ("ΣΩΤΗΡ ΤΟΥ ΚΟΣΜΟΥ","Σωτήρ"),
    ("ΤΟ ΒΑΠΤΙΣΜΑ ΤΟΥ ΙΗΣΟΥ","Βαπτ.ΙΧ"),
    ("Η ΜΕΤΑΜΟΡΦΩΣΗ","Μεταμ."),
    ("ΟΛΥΜΠΙΑΚΟΙ ΑΓΩΝΕΣ","Ολυμπ."),
    ("ΤΟ ΠΑΙΧΝΙΔΙ ΤΗΣ ΤΥΧΗΣ","τ.παιχ."),
    ("ΤΑ ΝΟΥΜΕΡΑ ΤΗΣ ΤΥΧΗΣ","τ.αρ."),
    ("ΚΑΛΗ ΕΠΙΤΥΧΙΑ","ευχή"),
    ("ΝΙΚΗ ΤΗΣ ΤΥΧΗΣ","νίκη"),
]

print(f"Σύνολο φράσεων: {len(PHRASES)}")

# Υπολογισμός λεξαρίθμων
lex_data = []
for phrase, label in PHRASES:
    m_val = milesian(phrase)
    p_nums = positional_nums(phrase)
    m_nums = milesian_nums_1_80(phrase)
    p_uniq = sorted(set(n for n in p_nums if 1 <= n <= 24))
    m_uniq = sorted(set(n for n in m_nums if 1 <= n <= 80))
    lex_data.append({
        'phrase': phrase, 'label': label,
        'milesian': m_val,
        'pos_nums': p_uniq,
        'mil_nums': m_uniq,
        'digital_root': digital_root(m_val),
    })

signals = []

def z_binom(obs_count, total, p_exp):
    if p_exp <= 0 or p_exp >= 1:
        return 0
    p_obs = obs_count / total
    return (p_obs - p_exp) / np.sqrt(p_exp*(1-p_exp)/total)

# Pre-compute appearance vectors with numpy for all phrases
# mat shape: (N, 81)
mat_bool = mat[:, 1:].astype(bool)  # (N, 80), index 0 = number 1

def hits_subset(nums_list):
    """Vectorized count of draws where all nums_list appear."""
    if not nums_list:
        return 0
    idx = [n-1 for n in nums_list]  # 0-indexed
    return int(mat_bool[:, idx].all(axis=1).sum())

# ═══════════════════════════════════════════════════════════════════
# T1: Φράση-ως-υποσύνολο (θέσης 1-24)
# ═══════════════════════════════════════════════════════════════════
print("\n" + "="*60)
print("T1: Φράση-ως-υποσύνολο (θέσης 1-24)")
print("="*60)

for ld in lex_data:
    nums = ld['pos_nums']
    k = len(nums)
    if k < 2 or k > 10:
        continue
    p_exp = comb(80-k, 20-k) / comb(80, 20)
    hits = hits_subset(nums)
    z = z_binom(hits, N, p_exp)
    if abs(z) >= 3.0:
        print(f"  {ld['phrase']:34s} k={k} z={z:+.2f} hits={hits} exp={p_exp*N:.1f}")
        signals.append(('word_phrase_subset', ld['phrase'], z,
                        f"phrase_subset k={k}"))

# ═══════════════════════════════════════════════════════════════════
# T2: Φράση-ως-υποσύνολο Μιλήσιο (1-80)
# ═══════════════════════════════════════════════════════════════════
print("\n" + "="*60)
print("T2: Φράση-ως-υποσύνολο Μιλήσιο (1-80)")
print("="*60)

for ld in lex_data:
    nums = ld['mil_nums']
    k = len(nums)
    if k < 2 or k > 10:
        continue
    p_exp = comb(80-k, 20-k) / comb(80, 20)
    hits = hits_subset(nums)
    z = z_binom(hits, N, p_exp)
    if abs(z) >= 3.0:
        print(f"  {ld['phrase']:34s} nums={nums} z={z:+.2f} hits={hits}")
        signals.append(('mil_phrase_subset', ld['phrase'], z,
                        f"milesian_subset k={k} nums={nums}"))

# ═══════════════════════════════════════════════════════════════════
# T3: Μερικό match ≥K/N γράμματα (θέσης)
# ═══════════════════════════════════════════════════════════════════
print("\n" + "="*60)
print("T3: Μερικό match ≥K/N γράμματα (5-12 θέσης)")
print("="*60)

for ld in lex_data:
    nums = ld['pos_nums']
    n_uniq = len(nums)
    if n_uniq < 5 or n_uniq > 12:
        continue
    idx = [n-1 for n in nums]
    col_sums = mat_bool[:, idx].sum(axis=1)  # per-draw count of phrase nums

    for k in [n_uniq, n_uniq-1]:
        if k < 3:
            continue
        p_exp = sum(
            comb(n_uniq, j) * comb(80-n_uniq, 20-j) / comb(80, 20)
            for j in range(k, min(n_uniq, 20)+1)
        )
        hits = int((col_sums >= k).sum())
        z = z_binom(hits, N, p_exp)
        if abs(z) >= 3.0:
            print(f"  {ld['phrase']:30s} ≥{k}/{n_uniq} z={z:+.2f} hits={hits}")
            signals.append(('partial_phrase', f"{ld['phrase']}≥{k}/{n_uniq}", z,
                            f"partial_match {k}/{n_uniq}"))

# ═══════════════════════════════════════════════════════════════════
# T4: Διορθ. Εκπλήρωση — EXACTLY k, υπόλοιπα στο i+1 (numpy)
# ═══════════════════════════════════════════════════════════════════
print("\n" + "="*60)
print("T4: Διορθ. Εκπλήρωση (EXACTLY k → υπόλοιπα επόμενη)")
print("="*60)

for ld in lex_data:
    nums = ld['pos_nums']
    n_total = len(nums)
    if n_total < 4 or n_total > 8:
        continue
    idx = np.array([n-1 for n in nums])  # 0-indexed

    # Indicator matrix for phrase numbers
    in_draw = mat_bool[:, idx]  # (N, n_total)
    in_curr = in_draw[:-1]  # (N-1, n_total)
    in_next = in_draw[1:]   # (N-1, n_total)
    count_curr = in_curr.sum(axis=1)  # (N-1,)

    for k in range(2, n_total):
        rest = n_total - k
        if rest < 2:
            continue
        exact_k = count_curr == k
        triggers = int(exact_k.sum())
        if triggers < 300:
            continue
        # Hit = all phrase nums covered by curr OR next
        covered = (in_curr[exact_k] | in_next[exact_k]).all(axis=1)
        hits = int(covered.sum())
        p_exp = comb(80-rest, 20-rest) / comb(80, 20)
        z = z_binom(hits, triggers, p_exp)
        if abs(z) >= 3.0:
            print(f"  {ld['phrase']:28s} k={k}+{rest} z={z:+.2f} "
                  f"hits={hits}/{triggers} (exp={p_exp:.5f})")
            signals.append(('fulfillment_fixed', f"{ld['phrase']}_k{k}+{rest}", z,
                            f"fulfillment_fixed {k}+{rest} of '{ld['phrase']}'"))

# ═══════════════════════════════════════════════════════════════════
# T5: Ένωση 2 συνεχόμενων — numpy vectorized
# ═══════════════════════════════════════════════════════════════════
print("\n" + "="*60)
print("T5: Ένωση 2 συνεχόμενων (draw_i ∪ draw_{i+1} ⊇ phrase)")
print("="*60)

for ld in lex_data:
    nums = ld['pos_nums']
    k = len(nums)
    if k < 4 or k > 14:
        continue
    idx = np.array([n-1 for n in nums])

    in_draw = mat_bool[:, idx]  # (N, k)
    union_covered = (in_draw[:-1] | in_draw[1:]).all(axis=1)  # (N-1,)
    hits = int(union_covered.sum())
    n_pairs = N - 1

    # Analytical p_exp using inclusion-exclusion
    p_exp = 0.0
    for j in range(k+1):
        sign = (-1)**j
        if 80 - j < 20:
            # Cannot draw 20 from fewer than 20 numbers → comb=0
            cont = 0.0
        else:
            cont = sign * comb(k, j) * (comb(80-j, 20) / comb(80, 20))**2
        p_exp += cont

    z = z_binom(hits, n_pairs, p_exp)
    if abs(z) >= 3.0:
        print(f"  {ld['phrase']:32s} k={k} z={z:+.2f} hits={hits} exp={p_exp*n_pairs:.1f}")
        signals.append(('union2_cover', ld['phrase'], z,
                        f"union2_cover k={k}"))

# ═══════════════════════════════════════════════════════════════════
# T6: Εβδομαδιαία bias — numpy vectorized
# ═══════════════════════════════════════════════════════════════════
print("\n" + "="*60)
print("T6: Εβδομαδιαία bias (φράση subset × ημέρα εβδομάδας)")
print("="*60)

for ld in lex_data:
    nums = ld['pos_nums']
    k = len(nums)
    if k < 2 or k > 8:
        continue
    p_overall = comb(80-k, 20-k) / comb(80, 20)

    idx = np.array([n-1 for n in nums])
    phrase_appears = mat_bool[:, idx].all(axis=1)  # (N,)

    for wd in range(7):
        mask = draw_weekdays == wd
        cnt = int(mask.sum())
        if cnt < 2000:
            continue
        hits_wd = int((phrase_appears & mask).sum())
        z = z_binom(hits_wd, cnt, p_overall)
        if abs(z) >= 3.5:
            print(f"  {ld['phrase']:30s} {WEEKDAY_NAMES[wd]} z={z:+.2f} "
                  f"hits={hits_wd}/{cnt}")
            signals.append(('phrase_weekday', f"{ld['phrase']}@wd{wd}", z,
                            f"phrase '{ld['phrase']}' bias on {WEEKDAY_NAMES[wd]}"))

# ═══════════════════════════════════════════════════════════════════
# T7: Ψηφιακή ρίζα αθροίσματος → αριθμός bias (z≥3.5)
# ═══════════════════════════════════════════════════════════════════
print("\n" + "="*60)
print("T7: Ψηφιακή ρίζα αθροίσματος → αριθμός bias (z≥3.5)")
print("="*60)

for dr in range(1, 10):
    mask = draw_dr == dr
    cnt = int(mask.sum())
    if cnt < 1000:
        continue
    sub_mat = mat[mask, 1:]  # (cnt, 80)
    rates = sub_mat.mean(axis=0)  # (80,)
    se = np.sqrt(0.25 * 0.75 / cnt)
    zscores = (rates - 0.25) / se
    sig = np.where(np.abs(zscores) >= 3.5)[0]
    for idx in sig:
        n = idx + 1
        z = float(zscores[idx])
        print(f"  dr={dr} #{n:2d} z={z:+.2f} (obs={rates[idx]:.4f}, n={cnt})")
        signals.append(('dr_bias', f"dr{dr}_n{n}", z,
                        f"digital_root_sum={dr} → #{n}"))

# ═══════════════════════════════════════════════════════════════════
# T8: Cross-draw resonance — φράση@draw_i × φράση@draw_{i+lag}
# ═══════════════════════════════════════════════════════════════════
print("\n" + "="*60)
print("T8: Cross-draw resonance (lag 1,2,3,5,10,17)")
print("="*60)

for ld in lex_data:
    nums = ld['pos_nums']
    k = len(nums)
    if k < 3 or k > 7:
        continue
    idx = np.array([n-1 for n in nums])
    appear = mat_bool[:, idx].all(axis=1).astype(np.int8)  # (N,)
    p_single = comb(80-k, 20-k) / comb(80, 20)
    p_exp_pair = p_single**2

    for lag in [1, 2, 3, 5, 10, 17]:
        hits = int(np.dot(appear[:-lag].astype(np.int32),
                          appear[lag:].astype(np.int32)))
        n_pairs = N - lag
        z = z_binom(hits, n_pairs, p_exp_pair)
        if abs(z) >= 3.5:
            print(f"  {ld['phrase']:30s} lag={lag:2d} z={z:+.2f} hits={hits}")
            signals.append(('cross_draw', f"{ld['phrase']}@lag{lag}", z,
                            f"cross_draw_resonance lag={lag}"))

# ═══════════════════════════════════════════════════════════════════
# T9: Back-projection — αν φράση σε draw_i, bias σε draw_{i-1} αριθμός
# ═══════════════════════════════════════════════════════════════════
print("\n" + "="*60)
print("T9: Back-projection (φράση@i → αριθμός bias@i-1)")
print("="*60)

for ld in lex_data:
    nums = ld['pos_nums']
    k = len(nums)
    if k < 3 or k > 7:
        continue
    idx = np.array([n-1 for n in nums])
    appear = mat_bool[:, idx].all(axis=1)  # (N,)

    # When phrase appears at i, look at draw i-1
    trigger_mask = appear[1:]  # phrase appears at index i = 1..N-1
    n_phrase = int(trigger_mask.sum())
    if n_phrase < 100:
        continue

    prev_mat = mat[:-1, 1:]  # (N-1, 80) — previous draws
    prev_when_phrase = prev_mat[trigger_mask]  # (n_phrase, 80)
    rates = prev_when_phrase.mean(axis=0)  # (80,)
    se = np.sqrt(0.25 * 0.75 / n_phrase)
    zscores = (rates - 0.25) / se
    sig = np.where(np.abs(zscores) >= 3.5)[0]
    for i_sig in sig:
        n = i_sig + 1
        z = float(zscores[i_sig])
        print(f"  {ld['phrase']:28s} → prev #{n:2d} z={z:+.2f} (n={n_phrase})")
        signals.append(('back_proj', f"{ld['phrase']}_prev_n{n}", z,
                        f"phrase@draw_i → #{n}@draw_{{i-1}}"))

# ═══════════════════════════════════════════════════════════════════
# T10: Ισοψηφία draw_sum = Μιλήσιος λεξαρίθμος
# ═══════════════════════════════════════════════════════════════════
print("\n" + "="*60)
print("T10: Ισοψηφία draw_sum = Μιλήσιος (range 400-1200)")
print("="*60)

sum_counts = Counter(draw_sums.tolist())
for ld in lex_data:
    v = ld['milesian']
    if v < 400 or v > 1200:
        continue
    obs = sum_counts.get(v, 0)
    p_exp = (1 / (90.1 * np.sqrt(2*np.pi))) * np.exp(-0.5*((v-810.1)/90.1)**2)
    exp = p_exp * N
    if exp < 1.0:
        continue
    z = (obs - exp) / np.sqrt(max(exp, 1))
    if abs(z) >= 3.0:
        print(f"  {ld['phrase']:30s} = {v} z={z:+.2f} obs={obs} exp={exp:.1f}")
        signals.append(('isopsephy_sum', f"sum={v}_{ld['phrase']}", z,
                        f"draw_sum={v} = milesian('{ld['phrase']}')"))

# ═══════════════════════════════════════════════════════════════════
# Αποτελέσματα
# ═══════════════════════════════════════════════════════════════════
print("\n" + "="*60)
print(f"ΣΥΝΟΛΟ ΣΗΜΑΤΩΝ BATCH 12: {len(signals)}")
print("="*60)
by_cat = defaultdict(list)
for cat, name, z, det in signals:
    by_cat[cat].append((abs(z), z, name, det))
for cat in sorted(by_cat.keys()):
    items = sorted(by_cat[cat], reverse=True)
    print(f"\n  [{cat}] — {len(items)} σήματα")
    for absz, z, name, det in items[:15]:
        print(f"    z={z:+.2f}  {name[:60]}")

# Save
out = {
    'batch': 'twelfth',
    'signals': [
        {'cat': cat, 'name': name, 'z': float(z), 'det': det}
        for cat, name, z, det in signals
    ]
}
with open('/home/user/Game/twelfth_signals.json', 'w') as f:
    json.dump(out, f, indent=2, ensure_ascii=False)
print(f"\nΑποθήκευση: twelfth_signals.json ({len(signals)} σήματα)")
