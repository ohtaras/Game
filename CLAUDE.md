# KINO App — Project Notes

## Repository
- **Repo**: `ohtaras/Game`
- **Live URL**: `https://ohtaras.github.io/Game/`
- **Pages source**: branch `gh-pages / (root)`
- **Deploy**: push to both `main` AND `gh-pages` (force) after every change

## Data
- **Location**: `data/raw/kino_raw_YYYY_MM.json` στο branch `main`
- **Range**: Ιανουάριος 2024 — Μάιος 2026 (29 αρχεία)
- **Κληρώσεις local**: ~236,810 (πριν την τελευταία ενημέρωση)
- **Πρώτη κλήρωση**: id=1062550 (Ιαν 2024)
- **Τελευταία κλήρωση local**: id=1299369 (Μάι 2026, πριν update +~290)
- **Μετά update**: ~237,100+ κληρώσεις (Μάι 2026 ενημερώθηκε επιτυχώς)
- **Format**: `{ month: "YYYY-MM", draws: [{id, n:[20 αριθμοί], b:bonus}] }`

## Αρχιτεκτονική app
- **`DATA_BRANCH = 'main'`** — branch για ανάγνωση/εγγραφή δεδομένων
- **`ghGet(path)`** — διαβάζει από `raw.githubusercontent.com/ohtaras/Game/main/` (CDN, γρήγορο, χωρίς auth). Fallback: GitHub API
- **`ghPut(path, obj, message)`** — γράφει με `branch: DATA_BRANCH` στο PUT body + retry 3x on 409
- **`getRawFileNames()`** — παράγει λίστα αρχείων Jan 2024→σήμερα (χωρίς API directory listing)
- **`knownRaw()` / `markKnown(filename)`** — localStorage cache για confirmed-existing αρχεία

## Grid & Κανόνες
- **Grid**: 8×10 toroidal, αριθμοί 1-80
- **Κάθετες στήλες**: `KINO_COLS` — 10 στήλες × 8 αριθμοί (col i = {i, i+10, ..., i+70})
- **OFFSET_RULES R1-R6**:
  - R1: offsets [6,8,33,38,45,61], ×5.4 (30 νίκες)
  - R2: offsets [28,31,39,45,47,78], ×4.7 (26 νίκες)
  - R3: offsets [1,17,29,30,58,64], ×4.3 (24 νίκες)
  - R4: offsets [22,23,48,63,66,67], ×4.3 (24 νίκες)
  - R5: offsets [6,7,12,16,26,56], ×4.2 (23 νίκες)
  - R6: offsets [0,3,24,32,35,51,73], ×15.3 (16 νίκες, 7-spot)
- **COL_PAYOUTS_8**: 4/8→€1, 5/8→€5, 6/8→€25, 7/8→€500, 8/8→€2500

## Γνωστά fixes που έγιναν
- Directory listing API (timeout) → αντικαταστάθηκε με `getRawFileNames()`
- SHA mismatch 409 → retry 3x + `branch: DATA_BRANCH` στο PUT body
- Branch με `/` στο όνομα δεν δουλεύει στο raw.githubusercontent.com → data μεταφέρθηκε στο `main`
- Auth: `token TOKEN` format (όχι `Bearer`) για αυτό το token
