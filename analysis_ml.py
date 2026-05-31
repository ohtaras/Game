"""
SCRIPT 4: Machine learning — can any model predict KINO draws above 25% baseline?

Memory-efficient design:
  One ROW per (draw_i, number_n) pair, but with COMPACT features:
    - own_delay[n]       : how many draws since n last appeared (capped at 60)  [1 feat]
    - own_lag1..lag5[n]  : did n appear in draw i-1 .. i-5                      [5 feats]
    - global_delay_mean  : mean delay across all 80 numbers at draw i            [1 feat]
    - global_delay_std   : std of delays                                         [1 feat]
    - local_hour         : Athens hour of draw i                                 [1 feat]
    - number_id          : n (1-80), so model can learn number-specific bias     [1 feat]
    TOTAL: 10 features per row  →  238K * 80 * 10 * 4 bytes ≈ 764 MB (fits!)

  Target: 1 if n appears in draw i+1

  Train/test split: first 80% draws → train, last 20% → test.
  Model: LogisticRegression (fast, interpretable).
  Evaluation: AUC-ROC, + top-8 pick strategies.

Anchor: id=1303293 → 2026-05-31 23:55 EEST = 2026-05-31 20:55 UTC
Minutes per draw = 5.28. DST: UTC+3 Apr-Sep, UTC+2 Oct-Mar.
"""

import json
import glob
import time
import numpy as np
from datetime import datetime, timezone

MINS_PER_DRAW = 5.28
ANCHOR_ID     = 1303293
ANCHOR_UTC    = datetime(2026, 5, 31, 20, 55, 0, tzinfo=timezone.utc)
DELAY_CAP     = 60
LAG_K         = 5
TRAIN_FRAC    = 0.80
N_FEAT        = 1 + LAG_K + 1 + 1 + 1 + 1   # = 10

# ── Load ──────────────────────────────────────────────────────────────────────
print("Loading draws...")
files = sorted(glob.glob("/home/user/Game/data/raw/kino_raw_*.json"))
draws = []
for f in files:
    with open(f) as fp:
        d = json.load(fp)
    draws.extend(d["draws"])
draws.sort(key=lambda x: x["id"])
N   = len(draws)
ids = np.array([d["id"] for d in draws], dtype=np.int64)
print(f"  {N:,} draws  |  ID range {ids[0]} – {ids[-1]}")

# ── Binary matrix ─────────────────────────────────────────────────────────────
print("Building presence matrix...")
M = np.zeros((N, 80), dtype=np.int8)
for i, d in enumerate(draws):
    for n in d["n"]:
        M[i, n - 1] = 1

# ── Athens local hour ────────────────────────────────────────────────────────
print("Computing local hours...")
anchor_epoch = ANCHOR_UTC.timestamp()
utc_epochs   = anchor_epoch + (ids - ANCHOR_ID) * MINS_PER_DRAW * 60.0
utc_dts      = [datetime.fromtimestamp(float(e), tz=timezone.utc) for e in utc_epochs]
months_a     = np.array([dt.month for dt in utc_dts], dtype=np.int8)
offset_a     = np.where((months_a >= 4) & (months_a <= 9), 3, 2)
utc_h_a      = np.array([dt.hour + dt.minute / 60.0 for dt in utc_dts])
local_h_a    = ((utc_h_a + offset_a) % 24.0).astype(np.float32)  # (N,)

# ── Build delay matrix (incremental) ─────────────────────────────────────────
# delay_mat[i, n] = draws since n last appeared BEFORE draw i, capped at DELAY_CAP
print("Computing delay matrix...")
START_I = LAG_K        # first draw with full lag history
END_I   = N - 2        # last draw where target i+1 exists
N_FEAT_DRAWS = END_I - START_I + 1   # number of usable draws

last_seen = np.full(80, -(DELAY_CAP + 1), dtype=np.int32)
# Warm up: process draws 0..START_I-1
for i in range(START_I):
    last_seen[np.where(M[i] == 1)[0]] = i

delay_mat = np.empty((N_FEAT_DRAWS, 80), dtype=np.int8)
t0 = time.time()
for ii, i in enumerate(range(START_I, END_I + 1)):
    raw_delay     = np.minimum(i - last_seen, DELAY_CAP)
    delay_mat[ii] = raw_delay.astype(np.int8)
    last_seen[np.where(M[i] == 1)[0]] = i
print(f"  delay_mat done in {time.time()-t0:.1f}s  shape={delay_mat.shape}")

# ── Build feature matrix and target ──────────────────────────────────────────
# N_FEAT = 10:
#   0       : own_delay[n] (how long since n appeared)
#   1-5     : own_lag_k[n] for k=1..5  (did n appear k draws ago)
#   6       : global_delay_mean  (mean delay of all 80 numbers)
#   7       : global_delay_std
#   8       : local_hour
#   9       : number_id (1-80, float)
#
# X has shape (N_FEAT_DRAWS * 80, 10) = ~19M * 10 * 4 bytes ≈ 764 MB
# y has shape (N_FEAT_DRAWS * 80,)   = ~19M * 1 byte

N_ROWS = N_FEAT_DRAWS * 80
print(f"\nFeature matrix: {N_FEAT_DRAWS:,} draws × 80 = {N_ROWS:,} rows × {N_FEAT} features")
print(f"  Memory estimate: {N_ROWS * N_FEAT * 4 / 1e9:.2f} GB (float32)")

t0 = time.time()

# Precompute draw-level scalars  (shape: N_FEAT_DRAWS)
draw_idx_range = np.arange(START_I, END_I + 1)
delay_f   = delay_mat.astype(np.float32)           # (N_FEAT_DRAWS, 80) float32
delay_mean = delay_f.mean(axis=1, keepdims=True)   # (N_FEAT_DRAWS, 1)
delay_std  = delay_f.std(axis=1, keepdims=True)    # (N_FEAT_DRAWS, 1)
hour_col   = local_h_a[draw_idx_range].reshape(-1, 1).astype(np.float32)  # (N_FEAT_DRAWS, 1)

# Lag features: for k=1..LAG_K, whether n appeared in draw i-k
# Shape: (N_FEAT_DRAWS, LAG_K, 80)
lag_3d = np.stack(
    [M[draw_idx_range - k].astype(np.float32) for k in range(1, LAG_K + 1)],
    axis=1
)  # (N_FEAT_DRAWS, LAG_K, 80)

# Target: did n appear in draw i+1?
target_mat = M[draw_idx_range + 1].astype(np.int8)   # (N_FEAT_DRAWS, 80)

# ── Flatten to (N_ROWS, N_FEAT) ───────────────────────────────────────────────
# For each draw ii, 80 rows (one per number).
# Row for number n at draw ii:
#   [delay_f[ii, n],
#    lag_3d[ii, 0, n], lag_3d[ii, 1, n], ..., lag_3d[ii, 4, n],
#    delay_mean[ii, 0],
#    delay_std[ii, 0],
#    hour_col[ii, 0],
#    float(n+1)]

# Build column by column (each column shape N_ROWS):
col_own_delay = delay_f.reshape(N_ROWS)                             # col 0
col_lags      = lag_3d.transpose(0, 2, 1).reshape(N_ROWS, LAG_K)  # cols 1-5
col_dmean     = np.repeat(delay_mean, 80, axis=0).reshape(N_ROWS)  # col 6
col_dstd      = np.repeat(delay_std,  80, axis=0).reshape(N_ROWS)  # col 7
col_hour      = np.repeat(hour_col,   80, axis=0).reshape(N_ROWS)  # col 8
col_numid     = np.tile(np.arange(1, 81, dtype=np.float32), N_FEAT_DRAWS)  # col 9

X = np.column_stack([
    col_own_delay,
    col_lags,
    col_dmean,
    col_dstd,
    col_hour,
    col_numid,
]).astype(np.float32)
y = target_mat.reshape(N_ROWS).astype(np.int8)

# Free intermediates
del delay_f, lag_3d, delay_mean, delay_std, hour_col
del col_own_delay, col_lags, col_dmean, col_dstd, col_hour, col_numid

print(f"  X shape: {X.shape}  y shape: {y.shape}  "
      f"({time.time()-t0:.1f}s build time)")
print(f"  Positive rate: {y.mean()*100:.3f}%  (expected 25.000%)")

# ── Train/test split ─────────────────────────────────────────────────────────
split_draw = int(N_FEAT_DRAWS * TRAIN_FRAC)
split_row  = split_draw * 80
X_train, X_test = X[:split_row],  X[split_row:]
y_train, y_test = y[:split_row],  y[split_row:]
n_test_draws    = N_FEAT_DRAWS - split_draw
del X, y

print(f"\nTrain: {len(X_train):,} rows ({split_draw:,} draws)")
print(f"Test : {len(X_test):,} rows ({n_test_draws:,} draws)")
print(f"Class balance train: {y_train.mean()*100:.3f}%  test: {y_test.mean()*100:.3f}%")

# ── Logistic Regression ───────────────────────────────────────────────────────
try:
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import roc_auc_score
    HAS_SKLEARN = True
except ImportError:
    HAS_SKLEARN = False
    print("sklearn not available — install with: pip install scikit-learn")

if HAS_SKLEARN:
    print("\nTraining LogisticRegression (solver=saga, max_iter=200)...")
    t0  = time.time()
    clf = LogisticRegression(
        max_iter=200,
        solver="saga",
        C=1.0,
        n_jobs=-1,
        verbose=1,
    )
    clf.fit(X_train, y_train)
    t_train = time.time() - t0
    print(f"  Training done in {t_train:.1f}s")

    t0    = time.time()
    y_prob = clf.predict_proba(X_test)[:, 1]  # P(appears in next draw)
    auc    = roc_auc_score(y_test, y_prob)
    print(f"  Inference + AUC done in {time.time()-t0:.1f}s")

    print("\n" + "=" * 74)
    print("LOGISTIC REGRESSION RESULTS")
    print("=" * 74)
    print(f"  AUC-ROC on test set       : {auc:.6f}")
    print(f"  Expected under H0 (random): 0.500000")
    print(f"  Lift above null           : {auc - 0.5:+.6f}")

    # ── Top-8 pick strategies ─────────────────────────────────────────────────
    # Reshape predictions and targets to (n_test_draws, 80)
    prob_mat  = y_prob.reshape(n_test_draws, 80)
    y_mat     = y_test.reshape(n_test_draws, 80).astype(np.int8)

    # Strategy 1: top-8 by model probability
    top8_model  = np.argsort(prob_mat, axis=1)[:, -8:]
    hits_model  = np.array([y_mat[i, top8_model[i]].sum() for i in range(n_test_draws)])

    # Strategy 2: top-8 by delay (longest-unseen = highest delay)
    # X_test rows 0, 80, 160, ... are the first number's row for each draw;
    # col 0 is own_delay for each number, and all 80 rows for a draw share the same
    # global context but differ in col 0 (own_delay).
    # Extract delay for each number per test draw:
    delay_test_mat = X_test[::80, 0].reshape(-1, 1)  # wrong — this is only num1's delay
    # Correct: reshape col 0 of X_test into (n_test_draws, 80)
    delay_test_mat = X_test[:, 0].reshape(n_test_draws, 80)  # col 0 = own_delay
    top8_delay  = np.argsort(delay_test_mat, axis=1)[:, -8:]
    hits_delay  = np.array([y_mat[i, top8_delay[i]].sum() for i in range(n_test_draws)])

    # Strategy 3: random 8 per draw
    rng          = np.random.default_rng(seed=42)
    random_picks = np.array([rng.choice(80, size=8, replace=False) for _ in range(n_test_draws)])
    hits_random  = np.array([y_mat[i, random_picks[i]].sum() for i in range(n_test_draws)])

    baseline_exp = 8 * 20 / 80   # = 2.0

    print(f"\n  Top-8 pick strategy comparison ({n_test_draws:,} test draws):")
    print(f"  {'Strategy':38}  {'AvgHits':>8}  {'vs_Baseline':>12}")
    print(f"  {'-'*65}")
    for label, hits in [
        ("Top-8 by model probability",       hits_model),
        ("Top-8 by delay (longest unseen)",  hits_delay),
        ("Random 8 (simulated)",             hits_random),
    ]:
        avg  = hits.mean()
        diff = avg - baseline_exp
        print(f"  {label:38}  {avg:>8.4f}  {diff:>+12.4f}")
    print(f"  {'Theoretical baseline (8×20/80=2.0)':38}  {baseline_exp:>8.4f}  {'0.0000':>12}")

    # ── Feature importances ───────────────────────────────────────────────────
    feat_names = (
        ["own_delay"]
        + [f"own_lag{k}" for k in range(1, LAG_K + 1)]
        + ["global_delay_mean", "global_delay_std", "local_hour", "number_id"]
    )
    coef     = clf.coef_[0]   # shape (N_FEAT,) = (10,)
    abs_coef = np.abs(coef)
    order    = np.argsort(abs_coef)[::-1]

    print(f"\n" + "=" * 74)
    print("FEATURE IMPORTANCES  (LogReg coefficients, all 10 features)")
    print("=" * 74)
    print(f"  {'Feature':25}  {'Coefficient':>13}  {'|Coef|':>10}")
    print(f"  {'-'*55}")
    for fi in order:
        print(f"  {feat_names[fi]:25}  {coef[fi]:>+13.6f}  {abs_coef[fi]:>10.6f}")

    # ── Verdict ───────────────────────────────────────────────────────────────
    print("\n" + "=" * 74)
    print("VERDICT")
    print("=" * 74)
    if auc > 0.52:
        print(f"  AUC = {auc:.6f} > 0.52  →  POSSIBLE SIGNAL DETECTED")
        print(f"  The model generalises slightly above chance.")
        print(f"  This may indicate mild structure (autocorrelation / delay effect).")
    else:
        print(f"  AUC = {auc:.6f} ≤ 0.52  →  No ML signal. Game is random.")
        print(f"  Logistic regression cannot predict KINO draws better than chance.")
        if auc > 0.505:
            print(f"  (Tiny lift {auc-0.5:+.4f} is likely due to the delay feature being a")
            print(f"   deterministic predictor for impossible events, not real signal.)")

else:
    print("\nInstall scikit-learn: pip install scikit-learn")

print("\nDone.")
