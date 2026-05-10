# KINO Data Fetcher

Εργαλείο για λήψη κληρώσεων KINO από το OPAP API και αποθήκευση στο GitHub.

## Γιατί τρέχει τοπικά

Το OPAP API επιτρέπει μόνο ελληνικές οικιακές IPs. Το script πρέπει να τρέχει από τον υπολογιστή σου.

## Εγκατάσταση

```bash
pip install requests
```

## Χρήση

```bash
# Θέσε το GitHub token σου
export GITHUB_TOKEN=ghp_xxxxxxxxxxxxxxxx

# Λήψη κληρώσεων για συγκεκριμένη ημερομηνία
python fetch_kino.py 2026-05-09

# Λήψη για σήμερα
python fetch_kino.py
```

Το script:
1. Κατεβάζει όλες τις κληρώσεις από το OPAP API
2. Αποθηκεύει τοπικά στο `kino/data/kino_YYYYMMDD.json`
3. Κάνει αυτόματο push στο GitHub repo `ohtaras/game`

## GitHub Token

Δημιούργησε token από: https://github.com/settings/tokens  
Απαιτούμενα permissions: `repo` (Contents: Read & Write)

## Μορφή δεδομένων

```json
{
  "date": "2026-05-09",
  "total_draws": 192,
  "draws": [
    {
      "draw_id": 1297020,
      "draw_time": "07:00",
      "winning_numbers": [3, 12, 17, ...],
      "bonus": []
    }
  ]
}
```
