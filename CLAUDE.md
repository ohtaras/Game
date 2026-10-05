# Game — Project Context for Claude

## Repo layout

| Branch | Purpose |
|--------|---------|
| `gh-pages` | Live app — HTML files served at `https://ohtaras.github.io/Game/` |
| `main` | Data files (JSON draws) |

**Always develop on `gh-pages` for HTML changes.** Never push to `main` unless explicitly updating data files.

## Pages

| File | Description |
|------|-------------|
| `ps_s3.html` | Main app — Power Spin + Super 3 + ΚΙΝΟ + Ωριαίο tabs |
| `kino_live.html` | Standalone ΚΙΝΟ Live page (new) |
| `kino_travel.html` | ΚΙΝΟ Χρονικό Ταξίδι |
| `kino_overlay.html` | ΚΙΝΟ Overlay |

## Games

### Power Spin (game ID `1110`)
- 3 wheels, values 1–27 each
- Draws every few minutes
- Data stored in `psDraws[]` — each entry: `{id, d, m, w:[w1,w2,w3]}`
  - `d`: Athens date string "YYYY-MM-DD"
  - `m`: minutes from midnight (Athens time)
  - `w`: wheel values

### Super 3 (game ID `1130`)
- 3 digits 0–9

### ΚΙΝΟ (game ID `1100`)
- 80 numbers, 20 drawn per draw
- Draws every **exactly 5 minutes**
- OPAP API returns `drawTime` as Unix timestamp ms (UTC)
- Monthly JSON format on `main` branch: `{id, n:[20 nums], b}` — **no date/time info**
- For date/time info always use the OPAP draws API directly

## OPAP API

```
https://api.opap.gr/draws/v3.0/{gameId}/draw-date/{date}/{date}?page=0&size=500
```

- Returns paginated JSON with `content[]`, `last` (boolean), `totalElements`
- `drawTime` = Unix timestamp ms UTC
- `winningNumbers.list[]` = drawn numbers
- Pagination: loop pages until `raw.last === true`; **no page limit** (API may return newest-first, so early morning draws appear on later pages)

## Athens timezone

```javascript
const ATHENS_MS = 3 * 3600000;           // UTC+3 fixed offset
const ath    = ms => new Date(ms + ATHENS_MS);
const dateOf = ms => { const d = ath(ms); return d.getUTCFullYear()+'-'+pad(d.getUTCMonth()+1)+'-'+pad(d.getUTCDate()); };
const minsOf = ms => { const d = ath(ms); return d.getUTCHours()*60 + d.getUTCMinutes(); };
```

Always use `dateOf()` / `minsOf()` for Athens-local date/time. Never use `toISOString()` (UTC) for local dates.

## ps_s3.html — key globals & functions

```javascript
psDraws[]          // all PS draws loaded so far
psFetchedDays      // Set<string> of already-fetched day strings
fetchDayLazy(ds)   // fetches PS+S3 for one day, marks psFetchedDays before fetch
autoFetchPs()      // polls for new PS draws; calls renderHourly() if hourly tab active
isHourlyActive()   // returns true if hourlyTab is visible
```

### Tab IDs
`ps`, `s3`, `kino`, `cmp`, `hot`, `hourly`, `cfg`

### Ωριαίο tab (hourly PS analysis)
- Shows last 30 days (sliding window, not calendar month)
- **Position-based slot matching**: group each day's draws within the selected hour by sequential position; use mode minute as button label. This handles the ~1–3 min variation in PS draw timing.
- Parallel fetch: 5 days at a time with `Promise.all`
- Live update: `autoFetchPs()` callback calls `renderHourly()` when tab is visible
- Prediction: leave-one-out cross-validation per slot AND whole-hour backtest panel
- `hourlyGoNow()` auto-selects current hour + last completed slot
- Next-slot label computed from ordered slot buttons (not from current slot)

## kino_live.html — architecture

### State machine
```javascript
histPool[]   // all historical draws: {id, arr, set, b, d, m}
liveDraw     // latest completed draw (today)
randDraw     // random historical draw for the UPCOMING 5-min slot
evalDraw     // previous randDraw — shown for evaluation after new live draw arrives
```

### Slot matching
```javascript
function slotMin(m) { return Math.round(m / 5) * 5; }  // normalize to nearest 5-min slot
// OPAP records drawTime slightly before the scheduled minute (e.g. 14:29:55 → m=869, slot=870)

function nextDrawMin() {
  // returns next multiple-of-5 minute in Athens time
  var now = ath(Date.now());
  var mins = now.getUTCHours()*60 + now.getUTCMinutes() + now.getUTCSeconds()/60;
  return Math.ceil(mins / 5) * 5;
}
```

### Random pick
- Filter `histPool` by `slotMin(d.m) === nextDrawMin()` AND `d.d !== today`
- Picks a random entry from matching draws

### Live sync
- `scheduleRefresh()`: waits `msToNextDraw() + 15s`, then retries every 15s up to 12 times (~3 min) until new draw appears
- `tickCd()`: detects slot change every second; re-picks `randDraw` immediately when `nextDrawMin()` changes

### History loading
- `loadToday()`: fetches today's draws first (fast) — sets initial `liveDraw`
- `loadHistory()`: fetches last 14 days in parallel batches of 5 — populates `histPool`
- **No page limit** in `fetchDay()` — loops all pages until `raw.last === true`

## SECURITY — NEVER VIOLATE

**The file `tk` (leaked GitHub token from old PR #11) must NEVER be:**
- Committed to any branch
- Extracted or read
- Ported to any new file

## Git workflow

```bash
git checkout gh-pages          # always work here for HTML
git add <files>
git commit -m "..."
git push -u origin gh-pages
```

Commit attribution (always include at end of commit message):
```
Co-Authored-By: Claude Sonnet 4.6 <noreply@anthropic.com>
Claude-Session: https://claude.ai/code/session_017aW6Xt3sHX8yeHbCg37xWg
```

## Navigation header in ps_s3.html

```html
<a class="nav-link" href="kino_travel.html">← KINO Χρονικό Ταξίδι</a>
<a class="nav-link" href="kino_overlay.html">🎰 ΚΙΝΟ Overlay</a>
<a class="nav-link" href="kino_live.html">🟢 ΚΙΝΟ Live</a>
```
