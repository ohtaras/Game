#!/usr/bin/env python3
"""
Batch 13: Εκτεταμένη αναζήτηση φράσεων (300+) με τα τεστ που έδωσαν αληθινά
σήματα στο batch 12: fulfillment_fixed, union2_cover, word_phrase_subset,
mil_phrase_subset.

Νέες κατηγορίες φράσεων:
  - Ορθόδοξες προσευχές (Πάτερ ημών, ΠιστεύωΑ, Αξιόν εστιν, ...)
  - Λειτουργικά ("Άγιος ο Θεός", "Κύριε ελέησον" παραλλαγές)
  - Αρχαία Ελληνική (Σοφοκλής, Όμηρος, Ησίοδος, στίχοι)
  - Μυθολογία (12 Άθλοι, Άργιες ονομασίες)
  - Σύγχρονη ελληνική κουλτούρα (τραγούδια, ποιήματα, τοπωνύμια)
  - Μαθηματικά / επιστήμη / αρχιτεκτονική
  - Νέες θεολογικές φράσεις (αγιολόγιο)
"""

import json, glob, re
from collections import defaultdict, Counter
from math import comb
import numpy as np

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
print(f"Κληρώσεις: {N:,}")

mat = np.zeros((N, 81), dtype=np.int8)
for i, d in enumerate(draws_raw):
    for n in d['n']:
        mat[i, n] = 1
mat_bool = mat[:, 1:].astype(bool)

# Λεξαρίθμοι
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

def positional_nums(text):
    return sorted(set(POSITIONAL[c] for c in normalize(text) if c in POSITIONAL))

def milesian_nums_1_80(text):
    return sorted(set(MILESIAN_1_80[c] for c in normalize(text) if c in MILESIAN_1_80))

def milesian(text):
    return sum(MILESIAN.get(c,0) for c in normalize(text))

# ═══════════════════════════════════════════════════════════════════
# Επεκταμένη λίστα φράσεων (300+)
# ═══════════════════════════════════════════════════════════════════
PHRASES = [
    # Ορθόδοξες προσευχές
    "ΠΑΤΕΡ ΗΜΩΝ", "ΕΛΘΕΤΩ Η ΒΑΣΙΛΕΙΑ", "ΓΕΝΗΘΗΤΩ ΤΟ ΘΕΛΗΜΑ ΣΟΥ",
    "ΑΓΙΑΣΘΗΤΩ ΤΟ ΟΝΟΜΑ", "ΤΟΝ ΑΡΤΟΝ ΗΜΩΝ", "ΑΦΕΣ ΗΜΙΝ", "ΑΦΕΣ ΗΜΙΝ ΤΑ ΟΦΕΙΛΗΜΑΤΑ",
    "ΡΥΣΑΙ ΗΜΑΣ ΑΠΟ ΤΟΥ ΠΟΝΗΡΟΥ", "ΠΕΙΡΑΣΜΟΝ",
    "ΠΙΣΤΕΥΩ ΕΙΣ ΕΝΑ ΘΕΟΝ", "ΘΕΟΝ ΠΑΤΕΡΑ", "ΠΟΙΗΤΗΝ ΟΥΡΑΝΟΥ ΚΑΙ ΓΗΣ",
    "ΦΩΣ ΕΚ ΦΩΤΟΣ", "ΘΕΟΝ ΑΛΗΘΙΝΟΝ", "ΟΜΟΟΥΣΙΟΝ ΤΩ ΠΑΤΡΙ",
    "ΑΞΙΟΝ ΕΣΤΙΝ", "ΩΣ ΑΛΗΘΩΣ ΜΑΚΑΡΙΖΕΙΝ",
    "ΘΕΟΤΟΚΕ ΠΑΡΘΕΝΕ", "ΧΑΙΡΕ ΚΕΧΑΡΙΤΩΜΕΝΗ", "Ο ΚΥΡΙΟΣ ΜΕΤΑ ΣΟΥ",
    "ΕΥΛΟΓΗΜΕΝΗ ΣΥ", "ΕΥΛΟΓΗΜΕΝΟΣ Ο ΚΑΡΠΟΣ",
    "ΑΓΙΟΣ Ο ΘΕΟΣ", "ΑΓΙΟΣ ΙΣΧΥΡΟΣ", "ΑΓΙΟΣ ΑΘΑΝΑΤΟΣ", "ΕΛΕΗΣΟΝ ΗΜΑΣ",
    "ΔΟΞΑ ΠΑΤΡΙ", "ΚΑΙ ΥΙΩ", "ΚΑΙ ΑΓΙΩ ΠΝΕΥΜΑΤΙ",
    "ΩΣ ΕΝ ΟΥΡΑΝΩ ΚΑΙ ΕΠΙ ΓΗΣ",
    "ΧΡΙΣΤΟΣ ΑΝΕΣΤΗ ΕΚ ΝΕΚΡΩΝ", "ΘΑΝΑΤΟΝ ΠΑΤΗΣΑΣ",
    "ΤΟΙΣ ΕΝ ΤΟΙΣ ΜΝΗΜΑΣΙ", "ΖΩΗΝ ΧΑΡΙΣΑΜΕΝΟΣ",

    # Λειτουργικές φράσεις
    "ΕΙΡΗΝΗ ΠΑΣΙ", "ΑΣ ΑΓΑΠΗΣΩΜΕΝ ΑΛΛΗΛΟΥΣ",
    "ΤΑ ΑΓΙΑ ΤΟΙΣ ΑΓΙΟΙΣ", "ΕΙΣ ΑΓΙΟΣ", "ΕΙΣ ΚΥΡΙΟΣ",
    "ΕΥΧΑΡΙΣΤΙΑ", "ΘΕΙΑ ΛΕΙΤΟΥΡΓΙΑ", "ΘΕΙΑ ΚΟΙΝΩΝΙΑ",
    "ΤΑ ΣΑ ΕΚ ΤΩΝ ΣΩΝ",
    "ΜΕΤΑΛΑΒΕ ΣΩΜΑ ΧΡΙΣΤΟΥ", "ΑΡΤΟΣ ΖΩΗΣ",
    "ΑΡΤΟΝ ΟΥΡΑΝΙΟΝ", "ΠΟΤΗΡΙΟΝ ΖΩΗΣ",

    # Αρχαία Ελληνική σοφία
    "ΕΥΡΗΚΑ", "ΑΡΧΙΜΗΔΗΣ",
    "ΟΙΔΑ ΟΥΔΕΝ ΕΙΔΩΣ", "ΣΩΚΡΑΤΗΣ", "ΠΛΑΤΩΝ", "ΑΡΙΣΤΟΤΕΛΗΣ",
    "ΟΜΗΡΟΣ", "ΗΡΑΚΛΕΙΤΟΣ", "ΠΑΡΜΕΝΙΔΗΣ", "ΔΗΜΟΚΡΙΤΟΣ",
    "ΕΠΙΚΟΥΡΟΣ", "ΖΗΝΩΝ", "ΧΡΥΣΙΠΠΟΣ",
    "ΘΑΛΗΣ", "ΑΝΑΞΙΜΑΝΔΡΟΣ", "ΑΝΑΞΙΜΕΝΗΣ", "ΞΕΝΟΦΑΝΗΣ",
    "ΗΣΙΟΔΟΣ", "ΠΙΝΔΑΡΟΣ",
    "ΕΣΧΑΤΟΣ ΛΟΓΟΣ", "ΕΝ ΤΟΥΤΩ ΝΙΚΑ",
    "ΟΥΚ ΕΣΤΙΝ ΑΛΛΟ", "ΕΞΑΡΓΥΡΩΣΙΣ",
    "ΕΥ ΖΗΝ", "ΕΥΔΑΙΜΟΝΙΑ", "ΕΥΤΥΧΙΑ",
    "ΑΘΑΝΑΣΙΑ", "ΘΝΗΤΟΣ", "ΑΘΑΝΑΤΟΣ",
    "ΘΕΩΡΙΑ", "ΠΡΑΞΙΣ", "ΑΛΗΘΗΣ ΛΟΓΟΣ",
    "ΑΡΕΤΗ", "ΚΑΚΙΑ", "ΕΥΓΕΝΕΙΑ",
    "ΚΑΛΛΟΣ", "ΟΡΟΣ ΟΛΥΜΠΟΣ",
    "ΚΕΡΑΥΝΟΣ ΤΟΥ ΔΙΟΣ", "ΑΙΓΙΣ",
    "ΗΡΑ ΖΕΥΣ", "ΕΣΤΙΑ",
    "ΗΦΑΙΣΤΟΣ", "ΗΡΑΚΛΗΣ", "ΘΗΣΕΥΣ", "ΠΕΡΣΕΥΣ", "ΟΔΥΣΣΕΥΣ",
    "ΑΧΙΛΛΕΥΣ", "ΑΓΑΜΕΜΝΩΝ", "ΕΛΕΝΗ", "ΠΑΡΙΣ",
    "ΕΚΤΩΡ", "ΑΙΑΣ", "ΔΙΟΜΗΔΗΣ",

    # Δωδεκάθεο τόπος
    "ΑΘΗΝΑ ΠΑΛΛΑΣ", "ΑΘΗΝΑ ΠΑΡΘΕΝΟΣ",
    "ΑΠΟΛΛΩΝ ΦΟΙΒΟΣ", "ΑΡΤΕΜΙΣ ΤΟΞΟΤΡΙΑ",
    "ΗΡΑ ΒΑΣΙΛΕΙΑ", "ΔΗΜΗΤΗΡ", "ΠΕΡΣΕΦΟΝΗ",
    "ΑΔΗΣ", "ΠΛΟΥΤΩΝ", "ΑΙΔΩΣ",
    "ΧΑΡΟΝ", "ΣΤΥΞ",
    "ΕΛΥΣΙΑ ΠΕΔΙΑ", "ΟΛΥΜΠΟΣ",
    "ΑΡΓΟΣ", "ΘΗΒΑΙ", "ΣΠΑΡΤΗ", "ΚΟΡΙΝΘΟΣ",

    # KINO / τυχαία (rerun verified ΤΖΑΚΠΟΤ etc.)
    "ΚΙΝΟ", "ΤΖΑΚΠΟΤ", "ΤΥΧΗ", "ΚΕΡΔΟΣ", "ΝΙΚΗ",
    "ΧΡΗΜΑ", "ΠΛΟΥΤΟΣ", "ΑΡΙΘΜΟΣ",
    "ΚΑΛΗ ΤΥΧΗ", "ΜΕΓΑΛΗ ΝΙΚΗ", "ΘΕΛΩ ΝΑ ΚΕΡΔΙΣΩ",
    "ΟΠΑΠ ΚΙΝΟ", "ΤΥΧΕΡΟΣ ΑΡΙΘΜΟΣ",
    "ΤΟ ΔΙΚΟ ΜΟΥ ΝΟΥΜΕΡΟ",

    # Νέα τεστ — ονόματα δωρητών, ευχών
    "ΓΕΙΑ ΣΑΣ", "ΚΑΛΗΣΠΕΡΑ", "ΚΑΛΗΜΕΡΑ", "ΚΑΛΗΝΥΧΤΑ",
    "ΣΤΗΝ ΥΓΕΙΑ", "ΣΤΗΝ ΥΓΕΙΑ ΜΑΣ",
    "ΧΡΟΝΙΑ ΠΟΛΛΑ", "ΕΥΧΑΡΙΣΤΩ",
    "ΣΥΓΧΑΡΗΤΗΡΙΑ", "ΣΥΛΛΥΠΗΤΗΡΙΑ",

    # Άγιοι ονομαστικά
    "ΑΓΙΟΣ ΒΑΣΙΛΕΙΟΣ", "ΑΓΙΟΣ ΣΠΥΡΙΔΩΝ",
    "ΑΓΙΟΣ ΧΡΥΣΟΣΤΟΜΟΣ",
    "ΑΓΙΟΣ ΠΑΥΛΟΣ", "ΑΓΙΟΣ ΠΕΤΡΟΣ",
    "ΑΓΙΟΣ ΑΝΔΡΕΑΣ", "ΑΓΙΟΣ ΣΤΕΦΑΝΟΣ",
    "ΑΓΙΑ ΕΛΕΝΗ", "ΑΓΙΑ ΑΙΚΑΤΕΡΙΝΗ",
    "ΑΓΙΑ ΠΑΡΑΣΚΕΥΗ", "ΑΓΙΑ ΑΝΑΣΤΑΣΙΑ",
    "ΑΓΙΟΣ ΠΑΝΤΕΛΕΗΜΟΝ",
    "ΑΓΙΟΣ ΚΟΣΜΑΣ", "ΑΓΙΟΣ ΔΑΜΙΑΝΟΣ",
    "ΑΓΙΟΣ ΧΑΡΑΛΑΜΠΟΣ", "ΑΓΙΟΣ ΕΛΕΥΘΕΡΙΟΣ",
    "ΑΓΙΟΣ ΜΑΡΚΟΣ", "ΑΓΙΟΣ ΛΟΥΚΑΣ",
    "ΑΓΙΟΣ ΜΑΤΘΑΙΟΣ",

    # Ιερά τοπία
    "ΑΓΙΟΝ ΟΡΟΣ", "ΑΓΙΟΣ ΤΑΦΟΣ",
    "ΙΕΡΟΥΣΑΛΗΜ", "ΒΗΘΛΕΕΜ", "ΝΑΖΑΡΕΤ",
    "ΣΙΝΑ ΟΡΟΣ", "ΘΑΒΩΡ",
    "ΓΕΘΣΗΜΑΝΗ", "ΓΟΛΓΟΘΑΣ",
    "ΚΩΝΣΤΑΝΤΙΝΟΥΠΟΛΗ", "ΣΜΥΡΝΗ",
    "ΕΦΕΣΟΣ", "ΑΛΕΞΑΝΔΡΕΙΑ",
    "ΠΑΝΑΓΙΑ ΣΟΥΜΕΛΑ", "ΤΗΝΟΣ",

    # Επιστήμη / Μαθηματικά
    "ΠΥΘΑΓΟΡΑΣ", "ΘΕΩΡΗΜΑ", "ΥΠΟΘΕΣΗ",
    "ΑΠΟΔΕΙΞΗ", "ΕΞΙΣΩΣΗ",
    "ΕΥΚΛΕΙΔΗΣ", "ΓΕΩΜΕΤΡΙΑ", "ΑΛΓΕΒΡΑ",
    "ΑΡΙΘΜΟΘΕΩΡΙΑ", "ΠΙΘΑΝΟΤΗΤΑ",
    "ΣΤΑΤΙΣΤΙΚΗ", "ΕΞΕΛΙΞΗ",
    "ΑΣΤΡΟΝΟΜΙΑ", "ΦΥΣΙΚΗ", "ΧΗΜΕΙΑ",
    "ΒΙΟΛΟΓΙΑ", "ΓΕΩΛΟΓΙΑ",
    "ΚΟΣΜΟΛΟΓΙΑ", "ΘΕΩΡΙΑ ΧΟΡΔΩΝ",

    # Σύγχρονα
    "ΕΛΛΑΔΑ", "ΑΘΗΝΑ ΠΟΛΗ",
    "ΕΛΛΗΝΙΚΗ ΔΗΜΟΚΡΑΤΙΑ",
    "ΑΣΦΑΛΕΙΑ", "ΟΙΚΟΓΕΝΕΙΑ",
    "ΦΙΛΟΙ ΚΑΙ ΟΙΚΟΓΕΝΕΙΑ",
    "ΧΑΡΑ ΚΑΙ ΥΓΕΙΑ",
    "ΟΝΕΙΡΑ ΓΛΥΚΑ",
    "ΑΓΑΠΗ ΖΩΗΣ", "ΕΡΩΤΑΣ ΖΩΗΣ",
    "ΓΑΛΗΝΗ", "ΗΣΥΧΙΑ",
    "ΕΥΛΟΓΙΑ", "ΕΥΧΑΡΙΣΤΙΑ ΣΤΟΝ ΘΕΟ",

    # Αρχιτεκτονική / μνημεία
    "ΠΑΡΘΕΝΩΝ", "ΕΡΕΧΘΕΙΟ",
    "ΠΡΟΠΥΛΑΙΑ", "ΘΗΣΕΙΟ",
    "ΝΑΟΣ ΤΟΥ ΔΙΟΣ", "ΟΡΑΚΛΕΙΟΥ ΣΥΜΒΟΥΛΕΥΤΗΡΙΟ",
    "ΑΓΟΡΑ", "ΣΤΟΑ",
    "ΘΕΑΤΡΟ ΤΟΥ ΔΙΟΝΥΣΟΥ",
    "ΩΔΕΙΟ ΗΡΩΔΟΥ ΤΟΥ ΑΤΤΙΚΟΥ",

    # Πανελλήνια τραγούδια / στίχοι
    "ΣΑΣ ΑΓΑΠΩ", "ΕΛΛΗΝΙΚΗ ΣΗΜΑΙΑ",
    "ΓΑΛΑΖΙΟ ΟΥΡΑΝΟ",
    "ΧΡΟΝΙΑ ΠΟΛΛΑ ΕΛΛΑΔΑ", "ΖΗΤΩ Η ΕΛΛΑΣ",
    "ΘΑΛΑΣΣΑ", "ΚΥΜΑΤΑ", "ΑΚΤΗ",
    "ΗΛΙΟΣ ΤΗΣ ΔΙΚΑΙΟΣΥΝΗΣ",

    # Νέα fulfillment-friendly (5-8 γράμματα)
    "ΑΝΘΟΣ", "ΦΩΛΕΑ", "ΣΠΟΡΟΣ",
    "ΡΙΖΑ", "ΚΛΑΔΟΣ", "ΦΥΛΛΟ",
    "ΚΑΡΠΟΣ", "ΔΕΝΔΡΟ",
    "ΦΩΤΙΑ", "ΑΕΡΑΣ", "ΝΕΡΟ",
    "ΓΗ", "ΟΥΡΑΝΟΣ",
    "ΛΙΒΑΝΟΣ", "ΣΜΥΡΝΑ",
    "ΧΡΥΣΟΣ", "ΑΡΓΥΡΟΣ", "ΧΑΛΚΟΣ",
    "ΣΙΔΗΡΟΣ", "ΜΟΛΥΒΔΟΣ",
    "ΡΟΔΟ", "ΚΡΙΝΟΣ", "ΒΙΟΛΕΤΑ",
    "ΛΕΩΝ", "ΤΑΥΡΟΣ", "ΛΥΚΟΣ",
    "ΑΕΤΟΣ", "ΓΕΡΑΚΙ", "ΠΕΡΙΣΤΕΡΑ",
    "ΨΑΡΙ", "ΔΕΛΦΙΝΙ",

    # Από Βίβλο / Αποκαλυπτικά
    "ΧΑΡΑΓΜΑ ΤΟΥ ΘΗΡΙΟΥ", "ΕΞΑΚΟΣΙΑ ΕΞΗΚΟΝΤΑ ΕΞ",
    "ΑΡΙΘΜΟΣ ΤΟΥ ΘΗΡΙΟΥ", "ΕΠΤΑ ΣΦΡΑΓΙΔΕΣ",
    "ΕΠΤΑ ΣΑΛΠΙΓΓΕΣ", "ΕΠΤΑ ΕΚΚΛΗΣΙΑΙ",
    "ΑΛΦΑ ΩΜΕΓΑ",
    "ΣΟΦΙΑ ΣΟΛΟΜΩΝΤΟΣ",
    "ΕΣΘΗΡ", "ΡΟΥΘ", "ΣΑΡΑ",

    # Σύντομες κρίσιμες
    "ΦΩΣ", "ΖΩΗ", "ΘΕΟΣ", "ΛΟΓΟΣ",
    "ΧΑΡΙΣ", "ΕΛΕΟΣ", "ΝΟΥΣ",
    "ΨΥΧΗ", "ΚΑΡΔΙΑ", "ΠΝΕΥΜΑ",
    "ΣΩΜΑ", "ΑΙΜΑ",
]

# Αφαίρεση διπλότυπων μετά κανονικοποίηση
seen_norm = set()
unique_phrases = []
for p in PHRASES:
    n = normalize(p)
    if n and n not in seen_norm:
        seen_norm.add(n)
        unique_phrases.append(p)

print(f"Φράσεις (μοναδικές): {len(unique_phrases)}")

# Υπολογισμός
lex_data = []
for p in unique_phrases:
    p_nums = positional_nums(p)
    m_nums = milesian_nums_1_80(p)
    lex_data.append({
        'phrase': p,
        'milesian': milesian(p),
        'pos_nums': p_nums,
        'mil_nums': m_nums,
    })

signals = []

def z_binom(obs, total, p_exp):
    if p_exp <= 0 or p_exp >= 1: return 0
    return (obs/total - p_exp) / np.sqrt(p_exp*(1-p_exp)/total)

def hits_subset(nums_list):
    if not nums_list: return 0
    idx = [n-1 for n in nums_list]
    return int(mat_bool[:, idx].all(axis=1).sum())

# ═══════════════════════════════════════════════════════════════════
# T1: word_phrase_subset (positional 1-24)
# ═══════════════════════════════════════════════════════════════════
print("\n" + "="*60)
print("T1: word_phrase_subset (positional 1-24)")
print("="*60)

for ld in lex_data:
    nums = ld['pos_nums']
    k = len(nums)
    if k < 4 or k > 9:
        continue
    p_exp = comb(80-k, 20-k) / comb(80, 20)
    if p_exp * N < 5:  # χρειαζόμαστε στατιστική ισχύ
        continue
    hits = hits_subset(nums)
    z = z_binom(hits, N, p_exp)
    if abs(z) >= 3.0:
        print(f"  {ld['phrase']:35s} k={k} nums={nums} z={z:+.2f} hits={hits} exp={p_exp*N:.1f}")
        signals.append({
            'cat': 'word_phrase_subset',
            'name': normalize(ld['phrase']),
            'z': float(z),
            'det': f"phrase_subset k={k} nums={nums} '{ld['phrase']}'",
            'nums': nums,
        })

# ═══════════════════════════════════════════════════════════════════
# T2: mil_phrase_subset (Μιλήσιο 1-80)
# ═══════════════════════════════════════════════════════════════════
print("\n" + "="*60)
print("T2: mil_phrase_subset (Milesian 1-80)")
print("="*60)

for ld in lex_data:
    nums = ld['mil_nums']
    k = len(nums)
    if k < 4 or k > 9:
        continue
    p_exp = comb(80-k, 20-k) / comb(80, 20)
    if p_exp * N < 5:
        continue
    hits = hits_subset(nums)
    z = z_binom(hits, N, p_exp)
    if abs(z) >= 3.0:
        print(f"  {ld['phrase']:35s} k={k} nums={nums} z={z:+.2f} hits={hits}")
        signals.append({
            'cat': 'mil_phrase_subset',
            'name': normalize(ld['phrase']) + '_mil',
            'z': float(z),
            'det': f"milesian_subset k={k} nums={nums} '{ld['phrase']}'",
            'nums': nums,
        })

# ═══════════════════════════════════════════════════════════════════
# T3: union2_cover (φράση καλύπτεται από draw_i ∪ draw_{i+1})
# ═══════════════════════════════════════════════════════════════════
print("\n" + "="*60)
print("T3: union2_cover (draw_i ∪ draw_{i+1} ⊇ phrase)")
print("="*60)

for ld in lex_data:
    nums = ld['pos_nums']
    k = len(nums)
    if k < 5 or k > 11:
        continue
    idx = np.array([n-1 for n in nums])
    in_draw = mat_bool[:, idx]
    cover = (in_draw[:-1] | in_draw[1:]).all(axis=1)
    hits = int(cover.sum())
    p_exp = sum((-1)**j * comb(k,j) * (comb(80-j,20)/comb(80,20))**2
                for j in range(k+1) if 80-j >= 20)
    if p_exp <= 0 or p_exp * (N-1) < 5:
        continue
    z = z_binom(hits, N-1, p_exp)
    if abs(z) >= 3.0:
        print(f"  {ld['phrase']:35s} k={k} z={z:+.2f} hits={hits} exp={p_exp*(N-1):.1f}")
        signals.append({
            'cat': 'union2_cover',
            'name': normalize(ld['phrase']),
            'z': float(z),
            'det': f"union2_cover k={k} nums={nums} '{ld['phrase']}'",
            'nums': nums,
        })

# ═══════════════════════════════════════════════════════════════════
# T4: fulfillment_fixed (EXACTLY k σε draw_i, υπόλοιπα σε draw_{i+1})
# ═══════════════════════════════════════════════════════════════════
print("\n" + "="*60)
print("T4: fulfillment_fixed (EXACTLY k → υπόλοιπα επόμενη)")
print("="*60)

for ld in lex_data:
    nums = ld['pos_nums']
    n_total = len(nums)
    if n_total < 4 or n_total > 9:
        continue
    idx = np.array([n-1 for n in nums])
    in_draw = mat_bool[:, idx]
    in_curr = in_draw[:-1]
    in_next = in_draw[1:]
    count_curr = in_curr.sum(axis=1)

    for k in range(2, n_total):
        rest = n_total - k
        if rest < 2:
            continue
        exact_k = count_curr == k
        triggers = int(exact_k.sum())
        if triggers < 500:
            continue
        covered = (in_curr[exact_k] | in_next[exact_k]).all(axis=1)
        hits = int(covered.sum())
        p_exp = comb(80-rest, 20-rest) / comb(80, 20)
        if p_exp * triggers < 5:
            continue
        z = z_binom(hits, triggers, p_exp)
        if abs(z) >= 3.0:
            print(f"  {ld['phrase']:32s} k={k}+{rest} z={z:+.2f} hits={hits}/{triggers}")
            signals.append({
                'cat': 'fulfillment_fixed',
                'name': normalize(ld['phrase']) + f'_k{k}+{rest}',
                'z': float(z),
                'det': f"EXACTLY {k} of {nums} in draw_i → remaining {rest} in draw_i+1 '{ld['phrase']}'",
                'nums': nums,
                'k': k, 'rest': rest,
            })

# ═══════════════════════════════════════════════════════════════════
# Αποτελέσματα
# ═══════════════════════════════════════════════════════════════════
print("\n" + "="*60)
print(f"ΣΥΝΟΛΟ BATCH 13: {len(signals)}")
print("="*60)
by_cat = defaultdict(list)
for s in signals:
    by_cat[s['cat']].append(s)
for cat in sorted(by_cat.keys()):
    items = sorted(by_cat[cat], key=lambda s: -abs(s['z']))
    print(f"\n  [{cat}] — {len(items)} σήματα")
    for s in items[:15]:
        print(f"    z={s['z']:+.2f}  {s['name'][:55]}")

with open('/home/user/Game/thirteenth_signals.json', 'w') as f:
    json.dump({'batch': 'thirteenth', 'signals': signals}, f, indent=2, ensure_ascii=False)
print(f"\nΑποθήκευση: thirteenth_signals.json ({len(signals)} σήματα)")
