#!/usr/bin/env python3
"""
Batch 11: Εκτεταμένη ανάλυση λεξαρίθμων — λέξεις ΚΑΙ φράσεις.

Συστήματα:
  A) Μιλήσιο (ιωνικό): Α=1,Β=2,Γ=3,Δ=4,Ε=5,Ζ=7,Η=8,Θ=9,Ι=10,Κ=20,Λ=30,
                         Μ=40,Ν=50,Ξ=60,Ο=70,Π=80,Ρ=100,Σ=200,Τ=300,Υ=400,
                         Φ=500,Χ=600,Ψ=700,Ω=800
  B) Θέσης (απλό):       Α=1...Ω=24

Τεστ:
  1. Φράση-ως-υποσύνολο (θέσης): όλοι οι αριθμοί-γράμματα στην ίδια κλήρωση
  2. Μερικό match: ≥K από N γράμματα βρίσκονται στην κλήρωση
  3. Άθροισμα κλήρωσης = τιμή Μιλήσιου λεξαρίθμου
  4. Αθροίσματα κλήρωσης ÷ τιμή λεξαρίθμου → τυχαία κατανομή;
  5. Αριθμός-στόχος από λεξαρίθμο mod 80
  6. Ημέρα εορτής × αριθμός (αγιολόγιο)
  7. Ψηφιακή ρίζα λεξαρίθμου (1-9) → bias
  8. Νέο: cross-draw resonance (φράση εμφανίζεται σε 2 συνεχόμενες κληρώσεις)
  9. Νέο: "εκπλήρωση" — αν K γράμματα σε κλήρωση_i, τα υπόλοιπα σε κλήρωση_i+1
"""

import json, glob, re
from collections import defaultdict, Counter
import numpy as np
from scipy import stats

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
print(f"Κληρώσεις: {N:,}")

# Δυαδικός πίνακας
mat = np.zeros((N, 81), dtype=np.int8)
for i, d in enumerate(draws_raw):
    for n in d['n']:
        mat[i, n] = 1

draw_sums = np.array([sum(d['n']) for d in draws_raw])
print(f"Sum range: {draw_sums.min()}-{draw_sums.max()}, mean={draw_sums.mean():.1f}")

# ═══════════════════════════════════════════════════════════════════
# Λεξαρίθμοι
# ═══════════════════════════════════════════════════════════════════

# Α) Μιλήσιο σύστημα
MILESIAN = {
    'Α':1,'Β':2,'Γ':3,'Δ':4,'Ε':5,'Ζ':7,'Η':8,'Θ':9,
    'Ι':10,'Κ':20,'Λ':30,'Μ':40,'Ν':50,'Ξ':60,'Ο':70,'Π':80,
    'Ρ':100,'Σ':200,'Τ':300,'Υ':400,'Φ':500,'Χ':600,'Ψ':700,'Ω':800,
    # Τελικό σίγμα
    'Ϲ':200,'ς':200,
}

# Β) Θέσης
GR_ORDER = 'ΑΒΓΔΕΖΗΘΙΚΛΜΝΞΟΠΡΣΤΥΦΧΨΩ'
POSITIONAL = {c: i+1 for i, c in enumerate(GR_ORDER)}

def normalize(text):
    """Μετατροπή σε κεφαλαία, αφαίρεση τόνων και μη-ελληνικών."""
    import unicodedata
    text = text.upper()
    # Αντικατάσταση τονισμένων
    replacements = {
        'Ά':'Α','Έ':'Ε','Ή':'Η','Ί':'Ι','Ό':'Ο','Ύ':'Υ','Ώ':'Ω',
        'Ϊ':'Ι','Ϋ':'Υ','ΐ':'Ι','ΰ':'Υ',
        'Ϲ':'Σ',  # σίγμα lunate
    }
    for a, b in replacements.items():
        text = text.replace(a, b)
    # Αφαίρεση ό,τι δεν είναι ελληνικό
    return re.sub(r'[^ΑΒΓΔΕΖΗΘΙΚΛΜΝΞΟΠΡΣΤΥΦΧΨΩ]', '', text)

def milesian(text):
    t = normalize(text)
    return sum(MILESIAN.get(c, 0) for c in t)

def positional_nums(text):
    """Αριθμοί 1-24 από γράμματα θέσης."""
    t = normalize(text)
    return [POSITIONAL[c] for c in t if c in POSITIONAL]

def digital_root(n):
    """Ψηφιακή ρίζα 1-9."""
    if n == 0: return 0
    return 1 + (n - 1) % 9

# ═══════════════════════════════════════════════════════════════════
# Φράσεις & λέξεις (150+)
# ═══════════════════════════════════════════════════════════════════

PHRASES = [
    # --- KINO / τύχη ---
    ("ΚΙΝΟ", "παιχνίδι"),
    ("ΤΥΧΗ", "τύχη"),
    ("ΚΑΛΗ ΤΥΧΗ", "ευχή"),
    ("ΚΕΡΔΟΣ", "κέρδος"),
    ("ΝΙΚΗ", "νίκη"),
    ("ΧΡΗΜΑ", "χρήμα"),
    ("ΠΛΟΥΤΟΣ", "πλούτος"),
    ("ΤΖΑΚΠΟΤ", "jackpot"),
    ("ΔΕΛΤΙΟ", "δελτίο"),
    ("ΑΡΙΘΜΟΣ", "αριθμός"),
    ("ΕΙΚΟΣΙ", "20"),
    ("ΟΓΔΟΝΤΑ", "80"),
    ("ΚΛΗΡΩΣΗ", "κλήρωση"),
    ("ΚΛΗΡΟΣ", "κλήρος"),
    ("ΕΚΤΑΚΤΗ ΤΥΧΗ", "έκτακτη τύχη"),

    # --- Θεολογικές φράσεις ---
    ("ΘΕΟΣ", "Θεός"),
    ("ΚΥΡΙΕ ΕΛΕΗΣΟΝ", "Κύριε ελέησον"),
    ("ΚΥΡΙΟΣ", "Κύριος"),
    ("ΙΗΣΟΥΣ ΧΡΙΣΤΟΣ", "Ιησούς Χριστός"),
    ("ΙΗΣΟΥΣ", "Ιησούς"),
    ("ΧΡΙΣΤΟΣ", "Χριστός"),
    ("ΧΡΙΣΤΟΣ ΑΝΕΣΤΗ", "Χριστός Ανέστη"),
    ("ΑΛΗΘΩΣ ΑΝΕΣΤΗ", "Αληθώς Ανέστη"),
    ("ΠΑΝΑΓΙΑ", "Παναγία"),
    ("ΘΕΟΤΟΚΟΣ", "Θεοτόκος"),
    ("ΜΑΡΙΑ", "Μαρία"),
    ("ΑΓΙΟΣ", "Άγιος"),
    ("ΑΓΙΑ ΤΡΙΑΣ", "Αγία Τριάς"),
    ("ΕΝ ΑΡΧΗ ΗΝ Ο ΛΟΓΟΣ", "Ιωάν. 1:1"),
    ("ΛΟΓΟΣ", "λόγος"),
    ("ΠΝΕΥΜΑ ΑΓΙΟΝ", "Πνεύμα Άγιον"),
    ("ΒΑΣΙΛΕΙΑ ΤΩΝ ΟΥΡΑΝΩΝ", "βασιλεία"),
    ("ΑΓΑΠΗ", "αγάπη"),
    ("ΠΙΣΤΙΣ", "πίστη"),
    ("ΕΛΠΙΣ", "ελπίδα"),
    ("ΣΟΦΙΑ", "σοφία"),
    ("ΑΛΗΘΕΙΑ", "αλήθεια"),
    ("ΖΩΗ", "ζωή"),
    ("ΦΩΣ", "φως"),
    ("ΣΚΟΤΟΣ", "σκότος"),
    ("ΑΓΓΕΛΟΣ", "άγγελος"),
    ("ΟΥΡΑΝΟΣ", "ουρανός"),
    ("ΠΑΡΑΔΕΙΣΟΣ", "παράδεισος"),
    ("ΑΜΗΝ", "αμήν"),
    ("ΑΛΛΗΛΟΥΙΑ", "αλληλούια"),
    ("ΔΟΞΑ", "δόξα"),
    ("ΔΟΞΑ ΤΩ ΘΕΩ", "δόξα τω Θεώ"),

    # --- Αρχαία Ελληνική σοφία ---
    ("ΓΝΩΘΙ ΣΑΥΤΟΝ", "Δελφοί"),
    ("ΜΗΔΕΝ ΑΓΑΝ", "Δελφοί"),
    ("ΕΝ ΟΙΔΑ ΟΤΙ ΟΥΔΕΝ ΟΙΔΑ", "Σωκράτης"),
    ("ΑΡΧΗ ΑΝΔΡΑ ΔΕΙΚΝΥΣΙ", "αρχή"),
    ("ΠΑΝΤΑ ΡΕΙ", "Ηράκλειτος"),
    ("ΛΟΓΟΣ ΤΟΥ ΠΑΝΤΟΣ", "λόγος παντός"),
    ("ΤΟ ΕΝ ΚΑΙ ΤΟ ΠΑΝ", "Ηράκλειτος"),
    ("ΑΡΙΘΜΟΣ ΕΣΤΙΝ Η ΑΡΧΗ", "Πυθαγόρας"),
    ("ΟΛΥΜΠΟΣ", "Όλυμπος"),
    ("ΖΕΥΣ", "Ζεύς"),
    ("ΑΘΗΝΑ", "Αθηνά"),
    ("ΑΠΟΛΛΩΝ", "Απόλλων"),
    ("ΕΡΜΗΣ", "Ερμής"),
    ("ΑΡΤΕΜΙΣ", "Άρτεμης"),
    ("ΠΟΣΕΙΔΩΝ", "Ποσειδών"),
    ("ΑΡΗΣ", "Άρης"),
    ("ΑΦΡΟΔΙΤΗ", "Αφροδίτη"),
    ("ΕΡΟΣ", "Έρως"),
    ("ΤΥΧΗ ΑΓΑΘΗ", "τύχη αγαθή"),
    ("ΚΑΛΟΣ ΚΑΓΑΘΟΣ", "καλοκαγαθία"),
    ("ΦΙΛΟΣΟΦΙΑ", "φιλοσοφία"),
    ("ΜΑΘΗΜΑΤΑ", "μαθήματα"),
    ("ΑΡΙΘΜΗΤΙΚΗ", "αριθμητική"),

    # --- Ελληνική ταυτότητα ---
    ("ΕΛΛΑΣ", "Ελλάς"),
    ("ΕΛΛΗΝ", "Έλλην"),
    ("ΕΛΛΗΝΙΣΜΟΣ", "ελληνισμός"),
    ("ΑΘΗΝΑ", "πόλη"),
    ("ΑΚΡΟΠΟΛΙΣ", "Ακρόπολη"),
    ("ΠΑΡΘΕΝΩΝ", "Παρθενώνας"),
    ("ΘΕΣΣΑΛΟΝΙΚΗ", "Θεσσαλονίκη"),
    ("ΚΡΗΤΗ", "Κρήτη"),
    ("ΚΥΠΡΟΣ", "Κύπρος"),
    ("ΙΘΑΚΗ", "Ιθάκη"),
    ("ΡΟΔΟΣ", "Ρόδος"),
    ("ΣΑΝΤΟΡΙΝΗ", "Σαντορίνη"),
    ("ΜΥΚΟΝΟΣ", "Μύκονος"),
    ("ΕΛΕΥΘΕΡΙΑ", "ελευθερία"),
    ("ΔΗΜΟΚΡΑΤΙΑ", "δημοκρατία"),
    ("ΠΑΤΡΙΔΑ", "πατρίδα"),
    ("ΕΘΝΟΣ", "έθνος"),

    # --- Αριθμολογία / εσωτερισμός ---
    ("ΕΠΤΑ", "7"),
    ("ΕΝΝΕΑ", "9"),
    ("ΔΩΔΕΚΑ", "12"),
    ("ΤΕΣΣΑΡΕΣΚΑΙΔΕΚΑ", "14"),
    ("ΕΙΚΟΣΙ", "20"),
    ("ΤΕΣΣΑΡΑΚΟΝΤΑ", "40"),
    ("ΕΒΔΟΜΗΚΟΝΤΑ", "70"),
    ("ΕΚΑΤΟΝ", "100"),
    ("ΧΙΛΙΟΙ", "1000"),
    ("ΑΛΦΑ ΚΑΙ ΩΜΕΓΑ", "Α+Ω"),
    ("ΑΛΦΑ", "Α"),
    ("ΩΜΕΓΑ", "Ω"),
    ("ΤΟ ΟΝΟΜΑ ΤΟΥ ΘΕΟΥ", "ΤΟΘΤ"),
    ("ΑΡΓΥΡΟΠΟΥΛΟΣ", "Αργυρόπουλος"),
    ("ΚΟΣΜΟΣ", "κόσμος"),
    ("ΣΥΜΠΑΝ", "σύμπαν"),
    ("ΑΙΩΝΙΟΤΗΤΑ", "αιωνιότητα"),
    ("ΕΙΜΑΡΜΕΝΗ", "ειμαρμένη"),
    ("ΜΟΙΡΑ", "μοίρα"),
    ("ΑΝΑΓΚΗ", "ανάγκη"),
    ("ΠΕΠΡΩΜΕΝΟ", "πεπρωμένο"),
    ("ΧΡΟΝΟΣ", "χρόνος"),
    ("ΑΙΩΝ", "αιών"),

    # --- Λεξαρίθμοι κλειδιά (ειδικές τιμές) ---
    ("ΑΔΑΜ", "Adam"),
    ("ΕΥΑ", "Eva"),
    ("ΝΟΗΣ", "Νώε"),
    ("ΑΒΡΑΑΜ", "Abraham"),
    ("ΜΩΥΣΗΣ", "Μωυσής"),
    ("ΔΑΒΙΔ", "Δαβίδ"),
    ("ΣΟΛΟΜΩΝ", "Σολομών"),
    ("ΙΩΑΝΝΗΣ", "Ιωάννης"),
    ("ΠΑΥΛΟΣ", "Παύλος"),
    ("ΠΕΤΡΟΣ", "Πέτρος"),
    ("ΓΕΩΡΓΙΟΣ", "Γεώργιος"),
    ("ΝΙΚΟΛΑΟΣ", "Νικόλαος"),
    ("ΔΗΜΗΤΡΙΟΣ", "Δημήτριος"),
    ("ΚΩΝΣΤΑΝΤΙΝΟΣ", "Κωνσταντίνος"),
    ("ΕΛΕΝΗ", "Ελένη"),

    # --- Φράσεις με ΕΓΚΡΥΠΤΑ νοήματα ---
    ("ΕΝ ΑΡΧ ΗΝ Ο ΛΟΓΟΣ", "Ιωάν 1:1 χωρίς Η"),
    ("ΤΟ ΦΩΣ ΤΟΥ ΚΟΣΜΟΥ", "Ιωάν 8:12"),
    ("ΕΓΩ ΕΙΜΙ Η ΟΔΟ", "Ιωάν 14:6"),
    ("ΑΓΑΠΑΤΕ ΑΛΛΗΛΟΥΣ", "Ιωάν 13:34"),
    ("ΟΙ ΕΣΧΑΤΟΙ ΕΣΟΝΤΑΙ ΠΡΩΤΟΙ", "Ματθ 20:16"),
    ("ΒΑΣΙΛΕΙΑ ΤΟΥ ΘΕΟΥ", "βασιλεία"),
    ("ΨΩΜΙ ΚΑΙ ΘΕΑΜΑΤΑ", "panem et circenses"),
    ("ΣΩΜΑ ΚΑΙ ΨΥΧΗ", "σώμα+ψυχή"),
    ("ΨΥΧΗ", "ψυχή"),
    ("ΣΩΜΑ", "σώμα"),
    ("ΝΟΥ ΚΑΙ ΚΑΡΔΙΑ", "νους+καρδιά"),
    ("ΝΟΥΣ", "νους"),
    ("ΚΑΡΔΙΑ", "καρδιά"),

    # --- Ημερολόγιο & φύση ---
    ("ΗΛΙΟΣ", "ήλιος"),
    ("ΣΕΛΗΝΗ", "σελήνη"),
    ("ΑΣΤΡΑ", "άστρα"),
    ("ΓΑΛΑΞΙΑΣ", "γαλαξίας"),
    ("ΣΕΙΡΙΟΣ", "Σείριος αστέρι"),
    ("ΠΑΝΣΕΛΗΝΟΣ", "πανσέληνος"),
    ("ΕΚΛΕΙΨΗ", "έκλειψη"),
    ("ΝΕΑ ΣΕΛΗΝΗ", "νέα σελήνη"),
    ("ΑΝΟΙΞΗ", "άνοιξη"),
    ("ΚΑΛΟΚΑΙΡΙ", "καλοκαίρι"),
    ("ΦΘΙΝΟΠΩΡΟ", "φθινόπωρο"),
    ("ΧΕΙΜΩΝΑΣ", "χειμώνας"),
    ("ΙΑΝΟΥΑΡΙΟΣ", "Ιανουάριος"),
    ("ΦΕΒΡΟΥΑΡΙΟΣ", "Φεβρουάριος"),
    ("ΜΑΡΤΙΟΣ", "Μάρτιος"),
    ("ΑΠΡΙΛΙΟΣ", "Απρίλιος"),
    ("ΜΑΙΟΣ", "Μάιος"),
    ("ΙΟΥΝΙΟΣ", "Ιούνιος"),
    ("ΙΟΥΛΙΟΣ", "Ιούλιος"),
    ("ΑΥΓΟΥΣΤΟΣ", "Αύγουστος"),
    ("ΣΕΠΤΕΜΒΡΙΟΣ", "Σεπτέμβριος"),
    ("ΟΚΤΩΒΡΙΟΣ", "Οκτώβριος"),
    ("ΝΟΕΜΒΡΙΟΣ", "Νοέμβριος"),
    ("ΔΕΚΕΜΒΡΙΟΣ", "Δεκέμβριος"),

    # --- Ειδικές αριθμολογικές αξίες ---
    ("ΑΜΗΝ", "99 — Μουσ. τιμή"),
    ("ΙΗΣ", "318 — Ιχθύς"),
    ("ΙΥΣ", "1480 — Ιησούς Μιλήσιο"),
]

print(f"\nΣύνολο φράσεων/λέξεων: {len(PHRASES)}")

# ═══════════════════════════════════════════════════════════════════
# Υπολογισμός λεξαρίθμων
# ═══════════════════════════════════════════════════════════════════
lex_data = []
for phrase, label in PHRASES:
    m_val = milesian(phrase)
    p_nums = positional_nums(phrase)
    p_nums_unique = sorted(set(n for n in p_nums if 1 <= n <= 80))
    dr = digital_root(m_val)
    lex_data.append({
        'phrase': phrase, 'label': label,
        'milesian': m_val,
        'pos_nums': p_nums_unique,
        'digital_root': dr,
        'mod80': m_val % 80 if m_val > 0 else 0,
    })

# ═══════════════════════════════════════════════════════════════════
# Τεστ 1: Φράση-ως-υποσύνολο (θέσης, αριθμοί 1-24)
# ═══════════════════════════════════════════════════════════════════
print("\n" + "="*60)
print("ΤΕΣΤ 1: Φράση-ως-υποσύνολο (όλοι οι αριθμοί θέσης ≤24)")
print("="*60)

signals = []

for ld in lex_data:
    nums = ld['pos_nums']
    if len(nums) < 2 or len(nums) > 10:
        continue  # πολύ σύντομο ή πολύ μακρύ

    k = len(nums)
    nums_set = set(nums)

    # P(set ⊆ draw) θεωρητικά (hypergeometric)
    # k αριθμοί όλοι ≤24, draw 20 από 80
    from math import comb
    p_expected = comb(80-k, 20-k) / comb(80, 20) if k <= 20 else 0

    # Παρατηρούμενο
    hits = sum(1 for s in M if nums_set <= s)
    p_obs = hits / N

    if p_expected > 0:
        z = (p_obs - p_expected) / np.sqrt(p_expected*(1-p_expected)/N)
        if abs(z) >= 2.3:
            print(f"  {ld['phrase']:30s} k={k} z={z:+.2f} "
                  f"obs={p_obs:.4f} exp={p_expected:.4f} ({hits} hits) [{ld['label']}]")
            if abs(z) >= 2.5:
                signals.append(('word_phrase_subset', ld['phrase'], z,
                                f"phrase subset (k={k}) {ld['phrase']}"))

# ═══════════════════════════════════════════════════════════════════
# Τεστ 2: Μερικό match — ≥K από N γράμματα φράσης
# ═══════════════════════════════════════════════════════════════════
print("\n" + "="*60)
print("ΤΕΣΤ 2: Μερικό match ≥K/N γράμματα (για φράσεις 5-12 γραμμάτων)")
print("="*60)

from math import comb

for ld in lex_data:
    nums = ld['pos_nums']
    n_letters = len(nums)
    if n_letters < 5 or n_letters > 12:
        continue

    nums_uniq = sorted(set(n for n in nums if 1 <= n <= 24))
    n_uniq = len(nums_uniq)
    if n_uniq < 4:
        continue

    # Δοκίμασε k = n_uniq και k = n_uniq-1
    for k in [n_uniq, n_uniq-1]:
        if k < 3:
            continue
        # P(≥k from nums_uniq in draw of 20 from 80)
        p_exp = sum(
            comb(n_uniq, j) * comb(80-n_uniq, 20-j) / comb(80, 20)
            for j in range(k, min(n_uniq, 20)+1)
        )
        hits = sum(1 for s in M if len(set(nums_uniq) & s) >= k)
        p_obs = hits / N
        if p_exp > 0:
            z = (p_obs - p_exp) / np.sqrt(p_exp*(1-p_exp)/N)
            if abs(z) >= 2.8:
                print(f"  {ld['phrase']:28s} ≥{k}/{n_uniq} z={z:+.2f} "
                      f"obs={p_obs:.4f} exp={p_exp:.4f}")
                if abs(z) >= 2.5:
                    signals.append(('partial_phrase', f"{ld['phrase']}≥{k}/{n_uniq}", z,
                                    f"partial match {k}/{n_uniq} of '{ld['phrase']}'"))

# ═══════════════════════════════════════════════════════════════════
# Τεστ 3: Άθροισμα κλήρωσης = τιμή Μιλήσιου λεξαρίθμου
# ═══════════════════════════════════════════════════════════════════
print("\n" + "="*60)
print("ΤΕΣΤ 3: draw_sum = Μιλήσιος λεξαρίθμος (range 419-1194)")
print("="*60)

sum_counts = Counter(draw_sums)

for ld in lex_data:
    v = ld['milesian']
    if v < 400 or v > 1200:
        continue
    obs = sum_counts.get(v, 0)
    # Expected: bin width 1 around mean 810, normal distribution
    p_exp = 1 / (90.1 * np.sqrt(2*np.pi)) * np.exp(-0.5*((v-810.1)/90.1)**2)
    exp = p_exp * N
    if exp < 0.5:
        continue
    z = (obs - exp) / np.sqrt(exp*(1-p_exp))
    if abs(z) >= 2.5:
        print(f"  {ld['phrase']:30s} = {v} z={z:+.2f} obs={obs} exp={exp:.1f} [{ld['label']}]")
        signals.append(('sum_eq_lex', f"sum={v}_{ld['phrase']}", z,
                        f"draw_sum={v} = {ld['phrase']} (Milesian)"))

# ═══════════════════════════════════════════════════════════════════
# Τεστ 4: Ψηφιακή ρίζα Μιλήσιου → αριθμός-στόχος bias
# ═══════════════════════════════════════════════════════════════════
print("\n" + "="*60)
print("ΤΕΣΤ 4: Ψηφιακή ρίζα Μιλήσιου λεξαρίθμου → αριθμός στόχος")
print("="*60)

# Ψηφιακή ρίζα αθροίσματος κλήρωσης → bias σε αριθμό
draw_dr = np.array([digital_root(int(s)) for s in draw_sums])  # 1-9

for dr in range(1, 10):
    mask = draw_dr == dr
    cnt = mask.sum()
    if cnt < 1000:
        continue
    for n in range(1, 81):
        p_obs = mat[mask, n].mean()
        p_exp = 0.25  # 20/80
        z = (p_obs - p_exp) / np.sqrt(p_exp*(1-p_exp)/cnt)
        if abs(z) >= 3.5:
            sign = '+' if z > 0 else '-'
            print(f"  dr={dr} #{n:2d} z={z:+.2f} (obs={p_obs:.4f} vs 0.25, n={cnt})")
            signals.append(('dr_bias', f"dr{dr}_n{n}", z,
                            f"digital_root={dr} → #{n}"))

# ═══════════════════════════════════════════════════════════════════
# Τεστ 5: Τιμή mod 80 → αριθμός στόχος
# ═══════════════════════════════════════════════════════════════════
print("\n" + "="*60)
print("ΤΕΣΤ 5: Μιλήσιος mod 80 → αριθμός στόχος")
print("="*60)

draw_mod80 = draw_sums % 80  # 0-79, target = mod+1 (1-80) or mod (0→80)

for ld in lex_data:
    v = ld['milesian']
    if v == 0:
        continue
    target = (v % 80)  # 0-79
    if target == 0:
        target = 80
    # Κληρώσεις όπου draw_sum mod 80 = v mod 80
    mask = draw_mod80 == (v % 80)
    cnt = mask.sum()
    if cnt < 500:
        continue
    p_obs = mat[mask, target].mean()
    p_exp = 0.25
    z = (p_obs - p_exp) / np.sqrt(p_exp*(1-p_exp)/cnt)
    if abs(z) >= 3.0:
        print(f"  {ld['phrase']:28s} mod80={v%80} → #{target} z={z:+.2f} (n={cnt})")
        signals.append(('mod80_target', f"lex_{ld['phrase']}_mod80_n{target}", z,
                        f"{ld['phrase']} mod80={v%80} → #{target}"))

# ═══════════════════════════════════════════════════════════════════
# Τεστ 6: "Εκπλήρωση" — K γράμματα τώρα + υπόλοιπα αύριο
# ═══════════════════════════════════════════════════════════════════
print("\n" + "="*60)
print("ΤΕΣΤ 6: Εκπλήρωση — K γράμματα σε draw_i, υπόλοιπα σε draw_i+1")
print("="*60)

for ld in lex_data:
    nums = sorted(set(n for n in ld['pos_nums'] if 1 <= n <= 24))
    n_total = len(nums)
    if n_total < 4 or n_total > 8:
        continue

    # Ψάξε: K γράμματα σε draw_i, υπόλοιπα (n_total-K) σε draw_{i+1}
    for k in range(2, n_total):
        rest = n_total - k
        if rest < 2:
            continue

        triggers = 0
        hits = 0
        from itertools import combinations
        nums_set = set(nums)
        for i in range(N-1):
            # Πόσα από nums είναι στην κλήρωση i
            in_i = nums_set & M[i]
            if len(in_i) >= k:
                # Ελέγξτε αν τα υπόλοιπα είναι στην κλήρωση i+1
                rest_nums = nums_set - M[i]
                if len(rest_nums) == rest and rest_nums <= M[i+1]:
                    hits += 1
                triggers += 1

        if triggers < 200:
            continue
        p_obs = hits / triggers
        # Expected: P(rest_nums ⊆ draw_{i+1})
        p_exp = comb(80-rest, 20-rest) / comb(80, 20) if rest <= 20 else 0
        if p_exp == 0:
            continue
        z = (p_obs - p_exp) / np.sqrt(p_exp*(1-p_exp)/triggers)
        if abs(z) >= 3.0:
            print(f"  {ld['phrase']:28s} k={k}+{rest} z={z:+.2f} "
                  f"hits={hits}/{triggers} exp_p={p_exp:.5f}")
            signals.append(('fulfillment', f"{ld['phrase']}_k{k}+{rest}", z,
                            f"fulfillment {k}+{rest} of '{ld['phrase']}'"))

# ═══════════════════════════════════════════════════════════════════
# Τεστ 7: Κατανομή αθρόισματος κλήρωσης mod λεξαρίθμου
# ═══════════════════════════════════════════════════════════════════
print("\n" + "="*60)
print("ΤΕΣΤ 7: Κατανομή draw_sum mod λεξαρίθμος (χ² ομοιομορφίας)")
print("="*60)

for ld in lex_data:
    v = ld['milesian']
    if v < 10 or v > 400:
        continue  # πολύ μικρό ή πολύ μεγάλο modulus

    remainders = draw_sums % v
    counts = np.bincount(remainders, minlength=v)
    expected = N / v
    if expected < 20:
        continue
    chi2 = np.sum((counts - expected)**2 / expected)
    df = v - 1
    z_chi2 = (chi2 - df) / np.sqrt(2*df)
    if abs(z_chi2) >= 3.5:
        print(f"  {ld['phrase']:28s} mod {v:4d} χ²z={z_chi2:+.2f} [{ld['label']}]")
        signals.append(('sum_mod_dist', f"sum_mod_{v}_{ld['phrase']}", z_chi2,
                        f"draw_sum mod {v} non-uniform (χ²z={z_chi2:.2f})"))

# ═══════════════════════════════════════════════════════════════════
# Τεστ 8: Ζεύγος φράσεων — ψηφιακή ρίζα product
# ═══════════════════════════════════════════════════════════════════
print("\n" + "="*60)
print("ΤΕΣΤ 8: Ζεύγη φράσεων — (A×B) mod 9 ψηφιακή ρίζα → bias")
print("="*60)

# Έλεγξε αν (mil_A × mil_B) mod 9 συσχετίζεται με draw_sum mod 9
# Αυτό τεστάρει αν κάποιοι λεξαρίθμοι έχουν "αρμονική" σχέση με τις κληρώσεις
draw_mod9 = draw_sums % 9
mod9_counts = np.array([np.sum(draw_mod9==r) for r in range(9)])
mod9_exp = N / 9

for i, (p1, l1) in enumerate(PHRASES[:30]):
    v1 = milesian(p1)
    if v1 == 0: continue
    dr1 = digital_root(v1)
    for p2, l2 in PHRASES[i+1:40]:
        v2 = milesian(p2)
        if v2 == 0: continue
        product_dr = digital_root(v1 * v2)
        target_mod9 = product_dr % 9
        obs_cnt = mod9_counts[target_mod9]
        z = (obs_cnt - mod9_exp) / np.sqrt(mod9_exp*(1-1/9))
        if abs(z) >= 3.5:
            print(f"  ({p1}) × ({p2}) → dr={product_dr} mod9={target_mod9} z={z:+.2f}")

# ═══════════════════════════════════════════════════════════════════
# Τεστ 9: Νέες συσχετίσεις — "ψυχή" του αριθμού
# Αν ο λεξαρίθμος ονόματος = άθροισμα αριθμών κλήρωσης mod 800
# ═══════════════════════════════════════════════════════════════════
print("\n" + "="*60)
print("ΤΕΣΤ 9: draw_sum mod Μιλήσιο → target number (500+ triggers)")
print("="*60)

for ld in lex_data:
    v = ld['milesian']
    if v < 2 or v > 800:
        continue

    # draw_sum mod v → target = mod result as number (if 1-80)
    remainders = draw_sums % v
    for target in range(1, 81):
        mask = remainders == target
        cnt = mask.sum()
        if cnt < 500:
            continue
        p_obs = mat[mask, target].mean()
        p_exp = 0.25
        z = (p_obs - p_exp) / np.sqrt(p_exp*(1-p_exp)/cnt)
        if abs(z) >= 3.5:
            print(f"  {ld['phrase']:25s} (v={v}) sum%{v}={target} → #{target} z={z:+.2f} n={cnt}")
            signals.append(('sum_mod_self', f"sum_mod{v}_n{target}_{ld['phrase']}", z,
                            f"draw_sum mod {v} = {target} → #{target}"))

# ═══════════════════════════════════════════════════════════════════
# Αποτελέσματα
# ═══════════════════════════════════════════════════════════════════
print("\n" + "="*60)
print(f"ΣΥΝΟΛΟ ΝΕΩΝ ΣΗΜΑΤΩΝ: {len(signals)}")
print("="*60)
for cat, name, z, det in sorted(signals, key=lambda x: -abs(x[2])):
    print(f"  {cat:20s} z={z:+.2f}  {name[:50]}")

# Save
out = {'batch': 'eleventh', 'signals': signals}
with open('/home/user/Game/eleventh_signals.json', 'w') as f:
    json.dump(out, f, indent=2, ensure_ascii=False)
print(f"\nΑποθήκευση: eleventh_signals.json")
