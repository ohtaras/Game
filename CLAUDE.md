# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

"Οικογενειακό Merge!" is a Greek-language family photo merge puzzle game. It is a **single-file static web app** (`index.html`) with no build toolchain, no package manager, and no test framework. All HTML, CSS, and JavaScript live in `index.html` (~736 lines).

## Running / Developing

Open `index.html` directly in a browser — no server required. Changes take effect on browser refresh. The app is deployed automatically to GitHub Pages on every push to `main` via `.github/workflows/deploy.yml`.

## Architecture

### Everything is in `index.html`

The file is structured in order: CSS styles → HTML markup → JavaScript. All game logic is plain vanilla JS with no external dependencies or module system.

### Game State (global variables)

```js
let grid = Array(16).fill(null);   // 4×4 board, each cell is null or { id }
let score, level, selected, lives; // core game state
let nextId, dropTimeout;           // pending tile and auto-drop timer handle
let paused, over;                  // game flow flags
let hiScores   // localStorage key 'mhi' – top-5 scores array
let customImages // localStorage key 'mci' – map of tile id → base64 data URL
```

### Tile System

Five tile types (`TILES` array, ids 1–5), each with a name (family member), embedded base64 photo, color theme, and point value (10/25/50/100/200). Players can replace default photos via the Setup screen; overrides are stored in `localStorage['mci']`.

### Level System

`LEVELS` array defines 5 levels by score threshold, weighted tile drop pool, and auto-drop interval (2200ms → 700ms). Tile pool weighting controls difficulty — lower-level tiles remain in the pool at higher levels.

### Core Loop

1. `dropTile()` places the `nextId` tile in a random empty cell and picks the next `nextId`.
2. `scheduleTimer()` sets a countdown; on expiry it calls `dropTile()` again.
3. `onCellClick(i)` handles selection → move (to empty) or merge (matching ids).
4. `merge(from, to)` upgrades the tile to `id+1`, awards points, checks level/game-over. Merging a max-level tile (id 5) removes it for 3× points.
5. Board full → `loseLife()` → clears 5 lowest tiles and shows the overlay. 3 lives total.

### Data Directory

`data/` holds Kino (Greek state lottery) draw history JSON files. These are **not used by the game UI** — they appear to be raw data for external analysis. `kino_master.json` (~22 MB) is the full history; daily files (`kino_YYYYMMDD.json`) and `kino_squares_master.json` are subsets/derived data.

### Images Directory

`images/` stores user-uploaded face photos (pattern `img*.jpg` is gitignored). A `.gitkeep` keeps the directory tracked. Default tile images are embedded as base64 directly in `index.html`.

## Security Note

The file `tk` in the repository root contains a GitHub Personal Access Token. **This token should be revoked immediately and the file removed** — it is a live credential committed to the repo.
