# BRIEFING ΓΙΑ ΝΕΟ AGENT — KINO Toroidal Square Research
> Τελευταία ενημέρωση: 2026-05-16

---

## 1. ΒΑΣΙΚΑ ΣΤΟΙΧΕΙΑ ΕΡΓΟΥ

| | |
|---|---|
| **GitHub Repo** | `ohtaras/Game` |
| **Feature branch** | `claude/setup-playground-repo-CTjhd` |
| **Deploy branch** | `gh-pages` |
| **Data branch** | `main` |
| **App URL** | `https://ohtaras.github.io/Game/` |
| **Local path** | `/home/user/Game/index.html` |
| **Git remote** | `http://local_proxy@127.0.0.1:40281/git/ohtaras/Game` |

**GitHub Token:** Αποθηκεύεται από τον χρήστη μέσω UI στο `localStorage['gh_token']`.
Ο agent δεν έχει direct access σε token — χρησιμοποιεί `mcp__github__*` tools ή το `ghPut/ghGet` του app.
Για Python scripts: χρησιμοποίησε `https://raw.githubusercontent.com/ohtaras/Game/main/...` για public reads (χωρίς token).

---

## 2. ΤΙ ΕΙΝΑΙ ΤΟ PROJECT

**KINO** = Ελληνικό λαχείο. Κάθε 5 λεπτά κληρώνονται 20 αριθμοί από το 1–80 (+ 1 bonus).

- Πριν Ιούν 2024: 192 κληρώσεις/ημέρα (08:00–23:55 EET, 16h)
- Μετά Ιούν 2024: 288 κληρώσεις/ημέρα (24h)

Η εφαρμογή:
1. Αποθηκεύει raw κληρώσεις στο GitHub (`data/raw/`)
2. Εντοπίζει **toroidal squares** σε κάθε κλήρωση
3. Αν εμφανιστεί ακριβώς 1 τετράγωνο → εφαρμόζει offset rules → ελέγχει αν τα προβλεπόμενα νούμερα εμφανίστηκαν στην αμέσως επόμενη (P1) ή μεθεπόμενη (P2) κλήρωση

---

## 3. ΔΟΜΗ ΔΕΔΟΜΕΝΩΝ

### Raw files — `data/raw/kino_raw_YYYY_MM.json`
```json
{
  "month": "2026-05",
  "draws": [
    {"id": 1298888, "n": [3,7,12,19,23,31,35,40,44,47,51,55,58,61,64,67,70,73,76,78], "b": 42},
    ...
  ]
}
```
- `id`: μοναδικό αύξον ID κλήρωσης
- `n`: 20 αριθμοί ταξινομημένοι αύξοντα (1–80)
- `b`: bonus αριθμός (null αν δεν υπάρχει)

**29 αρχεία:** `kino_raw_2024_01.json` → `kino_raw_2026_05.json`
**Σύνολο:** ~236,640 κληρώσεις (2024-02-14 → 2026-05-16)

### OPAP API
```
GET https://api.opap.gr/draws/v3.0/1100/draw-date/YYYY-MM-DD/YYYY-MM-DD?page=X&size=10
```
- Πάντα επιστρέφει **10 κληρώσεις/σελίδα** (αγνοεί το `size`)
- `raw.last` δεν υπάρχει (undefined) — break condition: `pg.length < 10`
- Μια ημέρα = max 29 σελίδες (288 / 10 = 28.8)
- Format απόκρισης: `{content: [...], last: bool, ...}` ή απευθείας array

### Draw ID math
```
BASE_ID = 1155022  → 2025-01-01 00:00 EET (= 2024-12-31 22:00 UTC)
BASE_MS = Date.UTC(2024, 11, 31, 22, 0, 0)  → 1735686000000 ms

dayN    = Math.floor((drawId - BASE_ID) / 288)
date    = new Date(BASE_MS + dayN * 86400000)
```

---

## 4. TOROIDAL SQUARES — ΑΛΓΟΡΙΘΜΟΣ

Το πλέγμα KINO είναι 8×10 (80 αριθμοί, γραμμές 0–7, στήλες 0–9).
Κάθε "toroidal square" είναι ένα 3×3 τετράγωνο +2 στήλες/γραμμές με wrap-around.

```python
def tor_sq_nums(root):
    r = (root - 1) // 10
    c = (root - 1) % 10
    return [
        root,                               # top-left
        r*10 + ((c+2) % 10) + 1,           # top-right  (+2 cols, wraps at 10)
        ((r+2) % 8)*10 + c + 1,            # bottom-left (+2 rows, wraps at 8)
        ((r+2) % 8)*10 + ((c+2) % 10) + 1  # bottom-right
    ]
    # 80 πιθανά squares (roots 1-80), κάθε ένα = 4 αριθμοί

def find_tor_squares(nums):
    s = set(nums)
    return [(n, tor_sq_nums(n)) for n in range(1, 81)
            if all(x in s for x in tor_sq_nums(n))]

def wrap_n(n):
    return ((n - 1) % 80) + 1
```

**Single-filter trigger:** Μια κλήρωση "ενεργοποιεί" κανόνα μόνο αν ακριβώς **1** toroidal square εμφανίζεται στα 20 νούμερά της.

---

## 5. OFFSET RULES (R1–R6)

```python
OFFSET_RULES = [
    {'id':1, 'label':'R1', 'offsets':[6,8,33,38,45,61]},     # 6 αριθμοί
    {'id':2, 'label':'R2', 'offsets':[28,31,39,45,47,78]},   # 6 αριθμοί
    {'id':3, 'label':'R3', 'offsets':[1,17,29,30,58,64]},    # 6 αριθμοί
    {'id':4, 'label':'R4', 'offsets':[22,23,48,63,66,67]},   # 6 αριθμοί
    {'id':5, 'label':'R5', 'offsets':[6,7,12,16,26,56]},     # 6 αριθμοί
    {'id':6, 'label':'R6', 'offsets':[0,3,24,32,35,51,73]},  # 7 αριθμοί
]

def get_preds(root, rule):
    return [wrap_n(root + d) for d in rule['offsets']]
```

**Νίκη:** Όλα τα προβλεπόμενα νούμερα εμφανίζονται στην P1 (επόμενη κλήρωση) **ή** P2 (μεθεπόμενη).
Ο έλεγχος γίνεται `all(pred in set(draw['n']) for pred in preds)`.

---

## 6. ΑΠΟΤΕΛΕΣΜΑΤΑ ΠΛΗΡΟΥΣ ΑΝΑΛΥΣΗΣ

**Dataset:** 236,640 κληρώσεις | 2024-02-14 → 2026-05-16 | ID: 1,062,550 – 1,299,199
**Single-trigger γεγονότα:** 42,878 (18.1% των κληρώσεων)

```
Κανόνας  Νίκες  Triggers  Win%     P1   P2   Avg_gap   Max_gap  Drought_τώρα
──────────────────────────────────────────────────────────────────────────────
R1         30    42,878   0.070%   11   19    1,448      9,292        96   ← μόλις χτύπησε
R2         26    42,878   0.061%   13   13    1,618      9,508     2,161
R3         24    42,878   0.056%   15    9    1,393      4,196     2,120
R4         24    42,878   0.056%   12   12    1,592      7,163     1,412
R5         23    42,878   0.054%   10   13    1,779      6,973     1,180
R6         16    42,878   0.037%    4   12    2,262      9,610     5,004  ← βαθύ drought
```

**Σημαντικά observations:**
- **R1** ισχυρότερος (0.070%), drought μόλις 96 triggers — μόλις χτύπησε
- **R3** πιο συνεπής: max_gap 4,196 (μικρότερο max από όλους), κλίνει προς P1
- **R6** σε βαθύ drought: 5,004 triggers (>52% του max 9,610)
- **R1** κλίνει προς P2 (11 P1, 19 P2 — χτυπά δηλαδή 2 κληρώσεις αργότερα)
- Μέσο κενό: ~1,400–2,300 triggers = ~7,000–11,500 κληρώσεις μεταξύ νικών

---

## 7. ΚΩΔΙΚΑΣ ΤΗΣ ΕΦΑΡΜΟΓΗΣ (index.html)

### Βασικές σταθερές (γραμμές ~302, ~657, ~1101, ~1182)
```javascript
const REPO = 'ohtaras/Game';

// Data update constants
const BASE_ID = 1155022;         // draw ID για 2025-01-01 00:00 EET
const BASE_MS = Date.UTC(2025, 0, 1);  // = 1735686000000 ms
const FETCH_FROM_YEAR = 2024;

// Live monitor constants
const LIVE_BASE_MS = Date.UTC(2024, 11, 31, 22, 0, 0);
const LIVE_BASE_ID = 1155022;

// Offset rules (γραμμές 1101-1108) — τα wins/x είναι ΠΑΛΙΑ, χρειάζονται update
const OFFSET_RULES = [
  {id:1, label:'R1', offsets:[6,8,33,38,45,61],    wins:26, x:'×3.9', color:'#e63946'},
  {id:2, label:'R2', offsets:[28,31,39,45,47,78],  wins:20, x:'×3.0', color:'#f4a261'},
  {id:3, label:'R3', offsets:[1,17,29,30,58,64],   wins:20, x:'×3.0', color:'#2a9d8f'},
  {id:4, label:'R4', offsets:[22,23,48,63,66,67],  wins:19, x:'×2.8', color:'#457b9d'},
  {id:5, label:'R5', offsets:[6,7,12,16,26,56],    wins:17, x:'×2.5', color:'#6a4c93'},
  {id:6, label:'R6', offsets:[0,3,24,32,35,51,73], wins:13, x:'×10.2',color:'#e76f51'},
];
```

### GitHub I/O (γραμμές 361–396)
```javascript
function ghHeaders() {
    return {
        'Authorization': 'token ' + localStorage.getItem('gh_token'),
        'Accept': 'application/vnd.github.v3+json',
        'Content-Type': 'application/json'
    };
}

async function ghGet(path) {
    // GitHub Contents API GET → auto-fallback για files >1MB (download_url)
    // επιστρέφει parsed JSON ή null
}

async function ghPut(path, obj, message) {
    // GitHub Contents API PUT (create ή update)
    // auto-fetches SHA για updates
    // timeout: 300000ms
}
```

### OPAP fetch (γραμμές 826–851)
```javascript
async function apiFetch(url) {
    // 3 fallbacks: direct → corsproxy.io → allorigins.win
}

async function fetchDaySeq(dayStr, label, size=10) {
    // paginate: page 0..35, break if pg.length < 10
    // επιστρέφει array of raw OPAP draw objects
}
```

### Core toroidal logic (γραμμές 880, 1110–1126)
```javascript
function wrapN(n) { return ((n-1)%80)+1; }
function torSqNums(root) {
    const r = Math.floor((root-1)/10), c = (root-1)%10;
    return [root, r*10+((c+2)%10)+1, ((r+2)%8)*10+c+1, ((r+2)%8)*10+((c+2)%10)+1];
}
function findTorSquares(nums) {
    const s = new Set(nums), found = [];
    for (let n = 1; n <= 80; n++) {
        const sq = torSqNums(n);
        if (sq.every(x => s.has(x))) found.push({root:n, sq});
    }
    return found;
}
function getOffsetPreds(root) {
    return OFFSET_RULES.map(r => ({...r, numbers: r.offsets.map(d => wrapN(root+d))}));
}
```

### Data update (γραμμές 643–767)
```javascript
async function updateRawByDay() {
    // Iterates months Jan 2024 → today
    // Skips complete past months (exist && !isCurrent)
    // Fetches missing months entirely
    // For current month: repairs days with <200 draws + fetches new days
    // Saves to data/raw/kino_raw_YYYY_MM.json on main branch via ghPut
    // 80ms delay between days to avoid rate limiting
}
```

---

## 8. ΕΚΚΡΕΜΟΤΗΤΕΣ

### α) Ενημέρωση OFFSET_RULES με σωστά στατιστικά (index.html γραμμή 1101)
```javascript
// ΠΑΛΙΑ → ΝΕΑ (από ανάλυση 236k κληρώσεων):
{id:1, label:'R1', offsets:[6,8,33,38,45,61],    wins:30, ...}
{id:2, label:'R2', offsets:[28,31,39,45,47,78],  wins:26, ...}
{id:3, label:'R3', offsets:[1,17,29,30,58,64],   wins:24, ...}
{id:4, label:'R4', offsets:[22,23,48,63,66,67],  wins:24, ...}
{id:5, label:'R5', offsets:[6,7,12,16,26,56],    wins:23, ...}
{id:6, label:'R6', offsets:[0,3,24,32,35,51,73], wins:16, ...}
```

### β) Διαγραφή παλιών αρχείων από GitHub `main`
Αχρείαστα αρχεία (μπορούν να διαγραφούν με `mcp__github__delete_file`):
- `data/kino_master.json` (22MB)
- `data/kino_20260503.json` έως `kino_20260511.json` (ημερήσια αρχεία)

### γ) Βαθύτερη ανάλυση των offset rules
Ερωτήματα για μελλοντική έρευνα:
- Υπάρχουν sub-patterns μέσα στα wins (π.χ. συγκεκριμένοι roots που νικούν περισσότερο);
- Τα P2 wins του R1 — τυχαία κατανομή ή clustering;
- Cross-rule: ποιες κληρώσεις έχουν win σε 2+ rules ταυτόχρονα;

---

## 9. GIT WORKFLOW

```bash
# Ανάπτυξη (code changes στο index.html):
git checkout claude/setup-playground-repo-CTjhd
# ... κάνε αλλαγές ...
git add index.html
git commit -m "Περιγραφή αλλαγής"
git push -u origin claude/setup-playground-repo-CTjhd
# → δημιούργησε draft PR προς main
```

Τα raw data αρχεία γράφονται **απευθείας στο `main`** μέσω GitHub API (`ghPut`).
ΔΕΝ χρησιμοποιούμε git για data files.

---

## 10. MCP GITHUB TOOLS (για agents)

Φόρτωσε πρώτα με `ToolSearch`:
```
ToolSearch("select:mcp__github__get_file_contents,mcp__github__push_files,...")
```

| Tool | Χρήση |
|---|---|
| `mcp__github__get_file_contents` | Διάβασε αρχείο από repo |
| `mcp__github__push_files` | Γράψε/ενημέρωσε αρχεία |
| `mcp__github__create_pull_request` | Δημιούργησε PR |
| `mcp__github__list_pull_requests` | Δες ανοιχτά PRs |
| `mcp__github__delete_file` | Διέγραψε αρχείο |
| `mcp__github__list_branches` | Λίστα branches |

**Scope:** Μόνο repo `ohtaras/Game`.

---

## 11. PYTHON TEMPLATE ΓΙΑ ΑΝΑΛΥΣΗ

```python
import urllib.request, json

def fetch(url):
    with urllib.request.urlopen(url, timeout=30) as r:
        return json.loads(r.read())

BASE = "https://raw.githubusercontent.com/ohtaras/Game/main/data/raw/"

# Κατέβασε όλα τα αρχεία
all_draws = []
for y in range(2024, 2027):
    for m in range(1, 13):
        if (y, m) > (2026, 5): break
        try:
            data = fetch(f"{BASE}kino_raw_{y}_{m:02d}.json")
            all_draws.extend(data['draws'])
            print(f"{y}-{m:02d}: {len(data['draws'])} draws")
        except Exception as e:
            print(f"{y}-{m:02d}: skip ({e})")

all_draws.sort(key=lambda d: d['id'])
print(f"Total: {len(all_draws)} draws")
# → ~236,640 draws

# Toroidal logic
def tor_sq_nums(root):
    r, c = (root-1)//10, (root-1)%10
    return [root, r*10+((c+2)%10)+1, ((r+2)%8)*10+c+1, ((r+2)%8)*10+((c+2)%10)+1]

def find_tor_squares(nums):
    s = set(nums)
    return [(n, tor_sq_nums(n)) for n in range(1,81) if all(x in s for x in tor_sq_nums(n))]

def wrap_n(n):
    return ((n-1) % 80) + 1

OFFSET_RULES = [
    {'id':1, 'label':'R1', 'offsets':[6,8,33,38,45,61]},
    {'id':2, 'label':'R2', 'offsets':[28,31,39,45,47,78]},
    {'id':3, 'label':'R3', 'offsets':[1,17,29,30,58,64]},
    {'id':4, 'label':'R4', 'offsets':[22,23,48,63,66,67]},
    {'id':5, 'label':'R5', 'offsets':[6,7,12,16,26,56]},
    {'id':6, 'label':'R6', 'offsets':[0,3,24,32,35,51,73]},
]

def get_preds(root, rule):
    return [wrap_n(root + d) for d in rule['offsets']]

# Main analysis loop
for i, draw in enumerate(all_draws[:-2]):
    sqs = find_tor_squares(draw['n'])
    if len(sqs) != 1:
        continue
    root = sqs[0][0]
    p1 = set(all_draws[i+1]['n'])
    p2 = set(all_draws[i+2]['n'])
    for rule in OFFSET_RULES:
        preds = set(get_preds(root, rule))
        p1_win = preds.issubset(p1)
        p2_win = preds.issubset(p2)
        # ... καταγραφή stats ...
```
