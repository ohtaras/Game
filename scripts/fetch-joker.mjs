// Server-side fetch of Joker (Τζόκερ) draws from OPAP, merged into data/raw/joker_raw.json.
// Joker draws 3x/week (Tue, Thu, Sun ~23:00 Athens time), so unlike Kino we keep all draws
// in one file instead of splitting by month.
//
// Run modes:
//   node scripts/fetch-joker.mjs            -> incremental: fetch only the latest draw
//   node scripts/fetch-joker.mjs --full      -> full backfill: paginate the entire history
// The OPAP product ID for Joker isn't documented publicly, so it's auto-discovered once
// (by probing candidate IDs against the known Joker draw shape: 5 numbers 1-45 + 1 bonus
// 1-20) and cached in data/joker_meta.json.

import fs from 'fs';
import path from 'path';

const DATA_DIR = path.join(process.cwd(), 'data', 'raw');
const META_FILE = path.join(process.cwd(), 'data', 'joker_meta.json');
const DATA_FILE = path.join(DATA_DIR, 'joker_raw.json');
const CANDIDATE_PRODUCT_IDS = [1101, 1102, 1103, 1104, 1105, 1106, 1107, 1108, 1109, 1110, 1111, 1112, 1113, 1114, 1115];

function isJokerShape(item) {
  const list = item?.winningNumbers?.list;
  const bonus = item?.winningNumbers?.bonus;
  return Array.isArray(list) && list.length === 5 && list.every(n => n >= 1 && n <= 45)
    && Array.isArray(bonus) && bonus.length === 1 && bonus[0] >= 1 && bonus[0] <= 20;
}

async function probeProductId(id) {
  try {
    const res = await fetch(`https://api.opap.gr/draws/v3.0/${id}/last-result-and-active`, { headers: { Accept: 'application/json' } });
    if (!res.ok) return false;
    const raw = await res.json();
    return isJokerShape(raw?.last);
  } catch { return false; }
}

async function resolveProductId() {
  if (fs.existsSync(META_FILE)) {
    const meta = JSON.parse(fs.readFileSync(META_FILE, 'utf8'));
    if (meta?.productId) return meta.productId;
  }
  for (const id of CANDIDATE_PRODUCT_IDS) {
    if (await probeProductId(id)) {
      fs.mkdirSync(path.dirname(META_FILE), { recursive: true });
      fs.writeFileSync(META_FILE, JSON.stringify({ productId: id }));
      console.log(`Discovered Joker product ID: ${id}`);
      return id;
    }
  }
  throw new Error('Could not discover the Joker OPAP product ID among candidates ' + CANDIDATE_PRODUCT_IDS.join(','));
}

function toRecord(item) {
  if (!isJokerShape(item)) return null;
  return { id: item.drawId, n: [...item.winningNumbers.list].sort((a, b) => a - b), b: item.winningNumbers.bonus[0] };
}

async function fetchRange(base, fromStr, toStr, size = 100) {
  let items = [], page = 0;
  while (true) {
    const url = `${base}/draw-date/${fromStr}/${toStr}?page=${page}&size=${size}`;
    let raw;
    try {
      const res = await fetch(url, { headers: { Accept: 'application/json' } });
      if (!res.ok) break;
      raw = await res.json();
    } catch (e) { console.error('fetch failed', fromStr, toStr, page, e.message); break; }
    const pg = Array.isArray(raw) ? raw : (raw.content || raw.draws || []);
    items = items.concat(pg);
    if (Array.isArray(raw) || raw.last === true || pg.length === 0 || pg.length < size) break;
    page++;
    if (page > 500) break;
  }
  return items;
}

function loadExisting() {
  if (!fs.existsSync(DATA_FILE)) return [];
  const data = JSON.parse(fs.readFileSync(DATA_FILE, 'utf8'));
  return data?.draws || [];
}

function saveDraws(draws) {
  fs.mkdirSync(DATA_DIR, { recursive: true });
  const sorted = [...draws].sort((a, b) => a.id - b.id);
  fs.writeFileSync(DATA_FILE, JSON.stringify({ draws: sorted }));
  return sorted.length;
}

async function runFull(base) {
  const existing = loadExisting();
  // Joker has existed since 1997; OPAP's modern draw IDs only need a wide range — the API
  // paginates by draw, not by literal date density, so one wide query is enough.
  const items = await fetchRange(base, '1997-01-01', new Date().toISOString().slice(0, 10));
  const fresh = items.map(toRecord).filter(Boolean);
  const merged = [...existing.filter(d => !fresh.some(f => f.id === d.id)), ...fresh];
  const total = saveDraws(merged);
  const newCount = merged.length - existing.length;
  console.log(`Full backfill: ${items.length} fetched, ${newCount} new, ${total} total.`);
}

async function runIncremental(base) {
  const existing = loadExisting();
  const last = existing[existing.length - 1];
  const res = await fetch(`${base}/last-result-and-active`, { headers: { Accept: 'application/json' } });
  if (!res.ok) throw new Error('HTTP ' + res.status);
  const raw = await res.json();
  const rec = toRecord(raw?.last);
  if (!rec) { console.log('No valid latest draw in response.'); return; }
  if (last && rec.id <= last.id) { console.log('No new draw.'); return; }
  const merged = saveDraws([...existing, rec]);
  console.log(`+1 new draw (#${rec.id}), ${merged} total.`);
}

async function main() {
  const productId = await resolveProductId();
  const base = `https://api.opap.gr/draws/v3.0/${productId}`;
  if (process.argv.includes('--full')) await runFull(base);
  else await runIncremental(base);
}

main().catch(e => { console.error(e); process.exit(1); });
