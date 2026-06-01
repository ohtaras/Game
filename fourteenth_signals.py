#!/usr/bin/env python3
"""
Batch 14: Συστηματική σάρωση — φράσεις-παραλλαγές με focus σε numbers
που έδωσαν σταθερά σήματα. Επίσης δοκιμή με όλους τους πιθανούς
συνδυασμούς ονομάτων αγίων και ορθόδοξων εορτών.
"""

import json, glob, re
from collections import defaultdict
from math import comb
import numpy as np

draws_raw = []
for f in sorted(glob.glob('/home/user/Game/data/raw/kino_raw_*.json')):
    with open(f) as fp: d = json.load(fp)
    draws_raw.extend(d['draws'])
draws_raw.sort(key=lambda x: x['id'])
N = len(draws_raw)
mat = np.zeros((N, 81), dtype=np.int8)
for i, d in enumerate(draws_raw):
    for n in d['n']: mat[i,n] = 1
mat_bool = mat[:, 1:].astype(bool)
print(f"Κληρώσεις: {N:,}")

GR = 'ΑΒΓΔΕΖΗΘΙΚΛΜΝΞΟΠΡΣΤΥΦΧΨΩ'
POS = {c:i+1 for i,c in enumerate(GR)}

def norm(t):
    t = t.upper()
    for a,b in {'Ά':'Α','Έ':'Ε','Ή':'Η','Ί':'Ι','Ό':'Ο','Ύ':'Υ','Ώ':'Ω',
                'Ϊ':'Ι','Ϋ':'Υ'}.items():
        t = t.replace(a,b)
    return re.sub(r'[^ΑΒΓΔΕΖΗΘΙΚΛΜΝΞΟΠΡΣΤΥΦΧΨΩ]','',t)

def pos_nums(t):
    return sorted(set(POS[c] for c in norm(t) if c in POS))

# Επεκταμένη λίστα — focus σε ονόματα, εορτές, λειτουργικά
PHRASES = [
    # Πάσχα / Ανάσταση
    "ΠΑΣΧΑ", "ΑΝΑΣΤΑΣΗ ΤΟΥ ΚΥΡΙΟΥ", "ΑΓΙΟΝ ΠΑΣΧΑ",
    "ΜΕΓΑΛΗ ΕΒΔΟΜΑΔΑ", "ΜΕΓΑΛΗ ΠΕΜΠΤΗ", "ΜΕΓΑΛΗ ΠΑΡΑΣΚΕΥΗ",
    "ΑΓΙΟΣ ΕΠΙΤΑΦΙΟΣ", "ΕΠΙΤΑΦΙΟΣ",
    "ΑΓΙΟΝ ΦΩΣ", "ΧΡΙΣΤΟΣ ΑΝΕΣΤΗ ΕΚ ΝΕΚΡΩΝ ΘΑΝΑΤΩ ΘΑΝΑΤΟΝ ΠΑΤΗΣΑΣ",

    # Χριστούγεννα
    "ΧΡΙΣΤΟΥΓΕΝΝΑ", "ΓΕΝΝΗΣΗ ΤΟΥ ΧΡΙΣΤΟΥ",
    "ΑΓΙΑ ΓΕΝΝΗΣΗ", "ΕΥΛΟΓΗΜΕΝΑ ΧΡΙΣΤΟΥΓΕΝΝΑ",
    "ΑΣΤΕΡΙ ΤΩΝ ΧΡΙΣΤΟΥΓΕΝΝΩΝ",

    # Επιφάνεια / Φώτα
    "ΘΕΟΦΑΝΕΙΑ", "ΦΩΤΑ", "ΑΓΙΑΣΜΟΣ",
    "ΒΑΠΤΙΣΗ ΤΟΥ ΧΡΙΣΤΟΥ",

    # Σταυρός
    "ΤΙΜΙΟΣ ΣΤΑΥΡΟΣ", "ΥΨΩΣΗ ΤΙΜΙΟΥ ΣΤΑΥΡΟΥ",
    "ΣΩΣΟΝ ΚΥΡΙΕ ΤΟΝ ΛΑΟΝ ΣΟΥ", "ΣΤΑΥΡΟΣ",

    # Νηστείες
    "ΣΑΡΑΚΟΣΤΗ", "ΚΥΡΙΑΚΗ ΤΗΣ ΟΡΘΟΔΟΞΙΑΣ",
    "ΚΑΘΑΡΑ ΔΕΥΤΕΡΑ", "ΑΠΟΚΡΙΕΣ",
    "ΝΗΣΤΕΙΑ ΧΡΙΣΤΟΥΓΕΝΝΩΝ",

    # Παναγία
    "ΓΕΝΕΣΗ ΤΗΣ ΘΕΟΤΟΚΟΥ", "ΕΙΣΟΔΟΣ ΤΗΣ ΘΕΟΤΟΚΟΥ",
    "ΕΥΑΓΓΕΛΙΣΜΟΣ ΘΕΟΤΟΚΟΥ",
    "ΚΟΙΜΗΣΗ ΘΕΟΤΟΚΟΥ", "ΔΕΚΑΠΕΝΤΑΥΓΟΥΣΤΟΣ",
    "ΥΠΕΡΑΓΙΑ ΘΕΟΤΟΚΕ ΣΩΣΟΝ ΗΜΑΣ",
    "ΘΕΟΤΟΚΕ ΠΑΡΘΕΝΕ ΧΑΙΡΕ",
    "ΑΓΝΗ ΠΑΡΘΕΝΕ ΔΕΣΠΟΙΝΑ",

    # 12 αποστόλοι
    "ΠΕΤΡΟΣ ΚΑΙ ΠΑΥΛΟΣ", "ΑΓΙΟΙ ΑΠΟΣΤΟΛΟΙ",
    "ΙΑΚΩΒΟΣ", "ΙΩΑΝΝΗΣ ΘΕΟΛΟΓΟΣ",
    "ΑΝΔΡΕΑΣ", "ΦΙΛΙΠΠΟΣ",
    "ΒΑΡΘΟΛΟΜΑΙΟΣ", "ΘΩΜΑΣ",
    "ΜΑΤΘΑΙΟΣ", "ΣΙΜΩΝ Ο ΖΗΛΩΤΗΣ",
    "ΘΑΔΔΑΙΟΣ", "ΙΟΥΔΑΣ ΙΣΚΑΡΙΩΤΗΣ",

    # Άλλες προσευχές
    "ΘΕΟΤΟΚΟΣ ΣΩΣΟΝ ΗΜΑΣ", "ΧΑΙΡΕΤΙΣΜΟΙ",
    "ΑΚΑΘΙΣΤΟΣ ΥΜΝΟΣ",
    "ΕΥΧΑΡΙΣΤΟΥΜΕΝ ΣΟΙ", "ΕΥΛΟΓΗΣΟΝ Ο ΘΕΟΣ",
    "ΣΥΓΧΩΡΗΣΟΝ ΗΜΑΣ",
    "ΑΝΑΣΤΗΘΗ Ο ΘΕΟΣ",

    # Ονόματα Θεού
    "ΑΛΦΑ ΚΑΙ ΩΜΕΓΑ Η ΑΡΧΗ ΚΑΙ ΤΟ ΤΕΛΟΣ",
    "Ο ΩΝ Ο ΗΝ Ο ΕΡΧΟΜΕΝΟΣ",
    "ΠΑΝΤΟΚΡΑΤΟΡ", "ΣΑΒΑΩΘ",
    "ΕΛΩΙ ΕΛΩΙ ΛΑΜΑ ΣΑΒΑΧΘΑΝΙ",

    # Άλλα ευλογιακά
    "ΧΡΟΝΙΑ ΠΟΛΛΑ ΚΑΙ ΕΥΛΟΓΗΜΕΝΑ",
    "ΚΑΛΕΣ ΓΙΟΡΤΕΣ",
    "ΟΛΟΨΥΧΑ", "ΟΛΟΨΥΧΕΣ ΕΥΧΕΣ",
    "ΟΛΟΚΑΡΔΙΑ", "ΑΓΑΠΗΜΕΝΑ ΠΑΙΔΙΑ",

    # Φυσικά φαινόμενα
    "ΗΛΙΑΚΗ ΕΚΛΕΙΨΗ", "ΣΕΛΗΝΙΑΚΗ ΕΚΛΕΙΨΗ",
    "ΠΑΝΣΕΛΗΝΟΣ ΑΥΓΟΥΣΤΟΥ",
    "ΑΣΤΡΑΠΗ", "ΒΡΟΝΤΗ", "ΟΥΡΑΝΙΟ ΤΟΞΟ",
    "ΑΥΓΗ", "ΛΥΚΟΦΩΣ", "ΔΕΙΛΙΝΟ",
    "ΛΑΜΠΡΗ ΑΣΤΡΟΦΕΓΓΙΑ",

    # Φιλοσοφία / Επιστήμη
    "ΠΥΘΑΓΟΡΕΙΟ ΘΕΩΡΗΜΑ",
    "ΧΡΥΣΗ ΤΟΜΗ", "ΧΡΥΣΟΣ ΑΡΙΘΜΟΣ", "ΦΙΟΝΑΤΣΙ",
    "ΑΡΙΘΜΟΣ ΠΙ", "ΕΞΙΣΩΣΗ ΕΥΛΕΡ",
    "ΑΠΟΛΥΤΟΣ ΑΡΙΘΜΟΣ", "ΦΥΣΙΚΟΣ ΑΡΙΘΜΟΣ",

    # Φιλοσοφικές αρχές
    "ΑΓΑΠΑ ΤΟΝ ΠΛΗΣΙΟΝ ΣΟΥ",
    "ΟΥΚ ΕΙΠΩΝ ΟΥΔΕΝ", "ΟΥΔΕΝ ΚΡΥΠΤΟΝ",
    "ΩΣ ΑΝΩ ΟΥΤΩ ΚΑΙ ΚΑΤΩ",
    "ΕΝ ΣΩΜΑΤΙ ΥΓΙΕΣ ΝΟΥΣ ΥΓΙΗΣ",

    # Νέα κατηγορία: ονόματα παλιά
    "ΣΟΦΟΚΛΗΣ", "ΕΥΡΙΠΙΔΗΣ", "ΑΙΣΧΥΛΟΣ",
    "ΑΡΙΣΤΟΦΑΝΗΣ", "ΘΟΥΚΥΔΙΔΗΣ", "ΗΡΟΔΟΤΟΣ",
    "ΞΕΝΟΦΩΝ", "ΔΗΜΟΣΘΕΝΗΣ",
    "ΠΕΡΙΚΛΗΣ", "ΛΕΩΝΙΔΑΣ",
    "ΑΛΕΞΑΝΔΡΟΣ Ο ΜΕΓΑΣ",
    "ΦΙΛΙΠΠΟΣ", "ΟΛΥΜΠΙΑΔΑ",
    "ΚΛΕΟΠΑΤΡΑ", "ΑΣΠΑΣΙΑ",

    # Φιλόσοφοι ευρύτερα
    "ΧΡΥΣΙΠΠΟΣ Ο ΣΟΛΕΥΣ",  # variant
    "ΠΛΟΥΤΑΡΧΟΣ", "ΕΠΙΚΤΗΤΟΣ",
    "ΜΑΡΚΟΣ ΑΥΡΗΛΙΟΣ", "ΣΕΝΕΚΑΣ",
    "ΠΟΡΦΥΡΙΟΣ", "ΠΛΩΤΙΝΟΣ",
    "ΠΡΟΚΛΟΣ", "ΙΑΜΒΛΙΧΟΣ",

    # 12 Ολυμπίων όλα
    "ΖΕΥΣ ΑΘΗΝΑ ΑΠΟΛΛΩΝ",
    "ΗΡΑ ΑΦΡΟΔΙΤΗ ΑΡΤΕΜΙΣ",
    "ΕΡΜΗΣ ΑΡΗΣ ΗΦΑΙΣΤΟΣ",
    "ΔΗΜΗΤΗΡ ΠΟΣΕΙΔΩΝ ΕΣΤΙΑ",

    # Ονόματα Ελλήνων μυθικά
    "ΟΡΦΕΥΣ", "ΕΥΡΥΔΙΚΗ",
    "ΙΑΣΩΝ", "ΜΗΔΕΙΑ",
    "ΘΗΣΕΥΣ ΑΡΙΑΔΝΗ",
    "ΜΙΝΩΤΑΥΡΟΣ", "ΛΑΒΥΡΙΝΘΟΣ",
    "ΔΑΙΔΑΛΟΣ", "ΙΚΑΡΟΣ",
    "ΠΡΟΜΗΘΕΥΣ", "ΕΠΙΜΗΘΕΥΣ", "ΠΑΝΔΩΡΑ",
    "ΑΤΛΑΣ", "ΧΑΡΩΝ",

    # Σύγχρονη Ελληνική
    "ΚΑΛΗ ΟΡΕΞΗ", "ΤΡΕΛΑ ΕΙΝΑΙ",
    "ΕΛΛΗΝΙΚΟ ΠΡΩΙ", "ΕΛΛΗΝΙΚΟ ΚΑΛΟΚΑΙΡΙ",

    # Ονομαστική γιορτή
    "ΤΑ ΠΟΛΛΑ", "ΕΥΧΕΣ ΓΙΑ ΟΛΑ",

    # Φράσεις από εθνικό ύμνο
    "ΣΕ ΓΝΩΡΙΖΩ ΑΠΟ ΤΗΝ ΚΟΨΗ",
    "ΟΣΤΕΑ ΙΕΡΑ ΤΩΝ ΕΛΛΗΝΩΝ",
    "ΣΑΛΠΙΓΓΕΣ ΤΗΣ ΕΛΕΥΘΕΡΙΑΣ",
]

# Αφαίρεση διπλότυπων
seen = set()
phrases = []
for p in PHRASES:
    n = norm(p)
    if n and n not in seen:
        seen.add(n)
        phrases.append(p)

print(f"Φράσεις: {len(phrases)}")

signals = []

def z_b(o, t, p):
    if p<=0 or p>=1: return 0
    return (o/t - p) / np.sqrt(p*(1-p)/t)

# T1: subset
for p in phrases:
    nums = pos_nums(p)
    k = len(nums)
    if k < 4 or k > 9: continue
    pe = comb(80-k, 20-k) / comb(80, 20)
    if pe * N < 5: continue
    idx = [n-1 for n in nums]
    h = int(mat_bool[:, idx].all(axis=1).sum())
    z = z_b(h, N, pe)
    if abs(z) >= 3.5:  # higher bar for multiple testing
        print(f"T1: {p:35s} k={k} {nums} z={z:+.2f} hits={h}")
        signals.append({'cat':'word_phrase_subset','name':norm(p),'z':float(z),
                        'det':f"phrase_subset k={k} nums={nums} '{p}'",'nums':nums})

# T2: union2
for p in phrases:
    nums = pos_nums(p)
    k = len(nums)
    if k < 5 or k > 11: continue
    idx = np.array([n-1 for n in nums])
    in_d = mat_bool[:, idx]
    cov = (in_d[:-1] | in_d[1:]).all(axis=1)
    h = int(cov.sum())
    pe = sum((-1)**j * comb(k,j) * (comb(80-j,20)/comb(80,20))**2
             for j in range(k+1) if 80-j>=20)
    if pe<=0 or pe*(N-1)<5: continue
    z = z_b(h, N-1, pe)
    if abs(z) >= 3.5:
        print(f"T2: {p:35s} k={k} z={z:+.2f} hits={h}")
        signals.append({'cat':'union2_cover','name':norm(p),'z':float(z),
                        'det':f"union2_cover k={k} nums={nums} '{p}'",'nums':nums})

# T3: fulfillment (faster, only most promising k)
for p in phrases:
    nums = pos_nums(p)
    n_t = len(nums)
    if n_t < 5 or n_t > 8: continue
    idx = np.array([n-1 for n in nums])
    in_d = mat_bool[:, idx]
    cnt = in_d[:-1].sum(axis=1)
    for k in [2, 3]:  # most useful k values
        rest = n_t - k
        if rest < 2 or rest > 6: continue
        ek = cnt == k
        trig = int(ek.sum())
        if trig < 500: continue
        cov = (in_d[:-1][ek] | in_d[1:][ek]).all(axis=1)
        h = int(cov.sum())
        pe = comb(80-rest, 20-rest) / comb(80, 20)
        if pe * trig < 5: continue
        z = z_b(h, trig, pe)
        if abs(z) >= 3.5:
            print(f"T3: {p:32s} k={k}+{rest} z={z:+.2f} hits={h}/{trig}")
            signals.append({'cat':'fulfillment_fixed','name':norm(p)+f'_k{k}+{rest}','z':float(z),
                            'det':f"EXACTLY {k} of {nums} → remaining {rest} in next '{p}'",
                            'nums':nums,'k':k,'rest':rest})

print(f"\nΣΥΝΟΛΟ BATCH 14: {len(signals)} signals (z≥3.5)")
for s in sorted(signals, key=lambda x: -abs(x['z'])):
    print(f"  z={s['z']:+.2f} [{s['cat']}] {s['name'][:50]}")

with open('/home/user/Game/fourteenth_signals.json','w') as f:
    json.dump({'batch':'fourteenth','signals':signals},f,indent=2,ensure_ascii=False)
print("\nΑποθήκευση: fourteenth_signals.json")
