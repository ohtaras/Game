#!/usr/bin/env python3
"""
Σειρά-επιπέδου ανάλυση: Markov chain σε clusters κληρώσεων.

1. Cluster all 238K draws into K=64 'topics' via k-means on 80-bit vectors
2. Compute transition matrix P(cluster_{t+1} | cluster_t)  — 64×64
3. Tests:
   a. Chi-square test for uniformity (baseline: random transitions)
   b. Top "predictive" transitions (z-score)
   c. Predictive accuracy: given cluster_t, how well predicts cluster_{t+1}?
   d. Perplexity vs uniform random model

This is the cleanest test for sequence-level memory in KINO.
"""
import json, time, numpy as np
from pathlib import Path
from sklearn.cluster import MiniBatchKMeans
from collections import Counter
from math import log, sqrt
from scipy import stats

DATA_DIR = Path('/home/user/Game/data/raw')

print("Loading draws...")
t0 = time.time()
all_draws = []
for f in sorted(DATA_DIR.glob('kino_raw_*.json')):
    with open(f) as fp:
        data = json.load(fp)
    for d in data.get('draws', []):
        all_draws.append((d['id'], sorted(d['n'])))
all_draws.sort(key=lambda x: x[0])
N = len(all_draws)
print(f"  {N} draws in {time.time()-t0:.1f}s")

# Build binary matrix
print("\nBuilding binary matrix (N × 80)...")
t0 = time.time()
M = np.zeros((N, 80), dtype=np.float32)
for i, (_, nums) in enumerate(all_draws):
    for n in nums:
        M[i, n-1] = 1.0
print(f"  Shape {M.shape}  in {time.time()-t0:.1f}s")

# ═══════════════════════════════════════════════════════════════════
# CLUSTER: k-means with K=64
# ═══════════════════════════════════════════════════════════════════
K = 64
print(f"\n══ Clustering into K={K} clusters (MiniBatchKMeans) ══")
t0 = time.time()
km = MiniBatchKMeans(n_clusters=K, batch_size=4096, max_iter=100,
                     random_state=42, n_init=3, verbose=0)
labels = km.fit_predict(M)
print(f"  Fit in {time.time()-t0:.1f}s")

# Cluster sizes
cluster_sizes = Counter(labels)
print(f"  Cluster sizes: min={min(cluster_sizes.values())} "
      f"max={max(cluster_sizes.values())} "
      f"mean={N/K:.0f}")
sizes = np.array([cluster_sizes[i] for i in range(K)])
print(f"  Coefficient of variation: {sizes.std()/sizes.mean():.3f}")

# ═══════════════════════════════════════════════════════════════════
# Transition matrix
# ═══════════════════════════════════════════════════════════════════
print(f"\n══ Building {K}×{K} transition matrix ══")
T = np.zeros((K, K), dtype=np.int64)
for i in range(N-1):
    T[labels[i], labels[i+1]] += 1

print(f"  Total transitions: {T.sum():,} (expected {N-1:,})")

# ═══════════════════════════════════════════════════════════════════
# TEST 1: Chi-square test for uniformity
# H0: P(j|i) = P(j) for all i  (i.e., no Markov memory, just marginal P(j))
# ═══════════════════════════════════════════════════════════════════
print(f"\n══ TEST 1: Chi-square — does cluster_t affect cluster_t+1? ══")

# Marginal probabilities
row_totals = T.sum(axis=1)
col_totals = T.sum(axis=0)
grand_total = T.sum()

# Expected under independence: E[i,j] = row_total[i] * col_total[j] / grand_total
expected = np.outer(row_totals, col_totals) / grand_total

# Chi-square
mask = expected > 5  # use only cells with E>5
chi2 = ((T[mask] - expected[mask])**2 / expected[mask]).sum()
df = (K-1)*(K-1) - (mask.size - mask.sum())  # degrees of freedom adjustment
p_value = 1 - stats.chi2.cdf(chi2, df)

print(f"  Chi-square statistic: {chi2:.2f}")
print(f"  Degrees of freedom: {df}")
print(f"  p-value: {p_value:.6f}")
if p_value < 0.001:
    print(f"  ★★★ HIGHLY SIGNIFICANT — sequence has memory!")
elif p_value < 0.05:
    print(f"  ★ Significant — possible memory")
else:
    print(f"  No significant memory — transitions appear random")

# ═══════════════════════════════════════════════════════════════════
# TEST 2: Top predictive transitions (z-score per cell)
# ═══════════════════════════════════════════════════════════════════
print(f"\n══ TEST 2: Top predictive transitions ══")
# For each cell, z = (T[i,j] - E[i,j]) / sqrt(E[i,j])
# (Poisson approximation)
with np.errstate(divide='ignore', invalid='ignore'):
    z_matrix = np.where(expected > 0, (T - expected) / np.sqrt(expected), 0)

# Top 15 positive
flat_idx = np.argsort(z_matrix.flatten())[::-1]
print(f"\n  Top 15 'predictive' transitions (highest +z):")
print(f"  {'i→j':>9} {'Observed':>9} {'Expected':>9} {'z':>7}")
for idx in flat_idx[:15]:
    i, j = idx // K, idx % K
    print(f"  {i:>3}→{j:<3}    {T[i,j]:>9} {expected[i,j]:>9.1f} {z_matrix[i,j]:>+7.2f}")

# Top 5 negative (avoided transitions)
print(f"\n  Top 5 'avoided' transitions (most negative z):")
for idx in flat_idx[-5:][::-1]:
    i, j = idx // K, idx % K
    print(f"  {i:>3}→{j:<3}    {T[i,j]:>9} {expected[i,j]:>9.1f} {z_matrix[i,j]:>+7.2f}")

# Distribution of |z| values
print(f"\n  Distribution of |z| across all {K*K} cells:")
abs_z = np.abs(z_matrix.flatten())
print(f"    max |z|: {abs_z.max():.2f}")
print(f"    mean |z|: {abs_z.mean():.2f}")
print(f"    cells with |z|>3: {(abs_z>3).sum()}  (expected by chance ~{K*K*0.0027:.0f})")
print(f"    cells with |z|>4: {(abs_z>4).sum()}  (expected by chance ~{K*K*0.0000633:.2f})")

# ═══════════════════════════════════════════════════════════════════
# TEST 3: Predictive accuracy
# Given cluster_t, predict argmax_j P(j|i). Test set accuracy.
# Split: first 80% train, last 20% test
# ═══════════════════════════════════════════════════════════════════
print(f"\n══ TEST 3: Predictive accuracy (next-cluster) ══")
split = int(N * 0.8)
train_labels = labels[:split]
test_labels = labels[split:]

# Transition matrix on training set
T_train = np.zeros((K, K), dtype=np.int64)
for i in range(len(train_labels)-1):
    T_train[train_labels[i], train_labels[i+1]] += 1

# Best next-cluster prediction per source cluster
best_next = T_train.argmax(axis=1)

# Test accuracy
correct = 0
for i in range(len(test_labels)-1):
    if best_next[test_labels[i]] == test_labels[i+1]:
        correct += 1
acc = correct / (len(test_labels)-1)
random_acc = 1.0 / K

# Best constant prediction (always guess most common cluster)
best_constant = T_train.sum(axis=0).argmax()
const_correct = sum(1 for l in test_labels[1:] if l == best_constant)
const_acc = const_correct / (len(test_labels)-1)

print(f"  Test set size: {len(test_labels):,}")
print(f"  Markov prediction accuracy:    {acc:.4f}")
print(f"  Always-most-common accuracy:   {const_acc:.4f}")
print(f"  Random baseline (1/K):         {random_acc:.4f}")
print(f"  Lift over random: {(acc/random_acc - 1)*100:+.1f}%")
print(f"  Lift over most-common: {(acc/const_acc - 1)*100:+.1f}%")

# ═══════════════════════════════════════════════════════════════════
# TEST 4: Perplexity comparison
# ═══════════════════════════════════════════════════════════════════
print(f"\n══ TEST 4: Perplexity (lower = better predictor) ══")
# Smoothed transition probs (Laplace +1)
T_smooth = T_train + 1
P_trans = T_smooth / T_smooth.sum(axis=1, keepdims=True)

# Markov log-likelihood
log_lik_markov = 0
for i in range(len(test_labels)-1):
    log_lik_markov += log(P_trans[test_labels[i], test_labels[i+1]])
perplexity_markov = np.exp(-log_lik_markov / (len(test_labels)-1))

# Marginal (no memory) baseline
marginal_counts = np.bincount(train_labels, minlength=K) + 1
P_marg = marginal_counts / marginal_counts.sum()
log_lik_marg = sum(log(P_marg[l]) for l in test_labels[1:])
perplexity_marg = np.exp(-log_lik_marg / (len(test_labels)-1))

# Uniform random
perplexity_unif = K

print(f"  Perplexity (uniform random): {perplexity_unif:.2f}")
print(f"  Perplexity (marginal):       {perplexity_marg:.2f}")
print(f"  Perplexity (Markov 1st-ord): {perplexity_markov:.2f}")
print(f"  Improvement Markov vs marg:  {(perplexity_marg/perplexity_markov - 1)*100:+.2f}%")
if perplexity_markov < perplexity_marg - 0.1:
    print(f"  ★ Markov captures real sequence info")
else:
    print(f"  Markov = marginal — no sequence info")

# ═══════════════════════════════════════════════════════════════════
# TEST 5: Higher-order Markov? (n-gram lag 2)
# ═══════════════════════════════════════════════════════════════════
print(f"\n══ TEST 5: Does lag-2 predict better than lag-1? ══")

# Compute log-likelihood with bigram (cluster_{t-1}, cluster_t) → cluster_{t+1}
# Use only training set; smooth heavily
print(f"  Building bigram model (sparse — using dict)...")
bigram_counts = {}
for i in range(len(train_labels)-2):
    key = (train_labels[i], train_labels[i+1])
    nxt = train_labels[i+2]
    if key not in bigram_counts:
        bigram_counts[key] = np.zeros(K)
    bigram_counts[key][nxt] += 1

# Test perplexity
log_lik_bi = 0
n_test = 0
for i in range(len(test_labels)-2):
    key = (test_labels[i], test_labels[i+1])
    nxt = test_labels[i+2]
    if key in bigram_counts:
        counts = bigram_counts[key] + 1  # Laplace
        p = counts[nxt] / counts.sum()
    else:
        # backoff to unigram
        p = P_marg[nxt]
    log_lik_bi += log(p)
    n_test += 1

perplexity_bi = np.exp(-log_lik_bi / n_test)
print(f"  Bigram perplexity: {perplexity_bi:.2f}")
print(f"  vs Unigram (Markov):   {perplexity_markov:.2f}")
print(f"  vs Marginal:           {perplexity_marg:.2f}")

print(f"\n══ ΤΕΛΟΣ MARKOV CLUSTERS ══")
print(f"\nΣυμπέρασμα:")
if p_value < 0.001 and perplexity_markov < perplexity_marg - 1:
    print("  ★ Υπάρχει σειρά-επιπέδου memory στο KINO!")
elif p_value < 0.05:
    print("  Πιθανή ασθενής δομή — όχι αξιοποιήσιμη")
else:
    print("  Καμία δομή — οι κληρώσεις είναι σειρά-ανεξάρτητες")
