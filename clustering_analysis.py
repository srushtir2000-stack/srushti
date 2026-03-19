"""
=============================================================================
Customer Churn Analysis for Telecommunications Company - Team 4
=============================================================================
Stage 2: Clustering Analysis
Task: Determine the Optimal Number of Clusters using the Elbow Method
Assigned to: Srushtiben Patel (CCAFTCT4-82 / CCAFTCT4-83)
=============================================================================
"""

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from kneed import KneeLocator
import warnings
warnings.filterwarnings('ignore')

# ── 1. Load the pre-processed, scaled data ──────────────────────────────────
X_train = pd.read_csv('/mnt/user-data/uploads/X_train_scaled.csv')
X_test  = pd.read_csv('/mnt/user-data/uploads/X_test_scaled.csv')

# Combine train + test for clustering (unsupervised – uses all available data)
X_all = pd.concat([X_train, X_test], axis=0, ignore_index=True)

print("=" * 70)
print("CLUSTERING ANALYSIS – ELBOW METHOD")
print("=" * 70)
print(f"\nDataset shape (combined): {X_all.shape}")
print(f"Features: {list(X_all.columns)}")
print(f"Number of samples: {X_all.shape[0]}")
print(f"Number of features: {X_all.shape[1]}")

# ── 2. Run K-Means for K = 1 to 15 and record WCSS (inertia) ───────────────
K_range = range(1, 16)
inertia_values = []
results = []

print("\n" + "-" * 70)
print(f"{'K':>4} | {'Inertia (WCSS)':>18} | {'Delta':>14} | {'% Reduction':>14}")
print("-" * 70)

prev_inertia = None
for k in K_range:
    kmeans = KMeans(
        n_clusters=k,
        init='k-means++',
        n_init=10,
        max_iter=300,
        random_state=42
    )
    kmeans.fit(X_all)
    inertia = kmeans.inertia_
    inertia_values.append(inertia)
    
    delta = (prev_inertia - inertia) if prev_inertia is not None else 0
    pct   = (delta / prev_inertia * 100) if prev_inertia is not None else 0
    
    results.append({
        'K': k,
        'Inertia': round(inertia, 2),
        'Delta': round(delta, 2),
        'Pct_Reduction': round(pct, 2)
    })
    
    print(f"{k:>4} | {inertia:>18.2f} | {delta:>14.2f} | {pct:>13.2f}%")
    prev_inertia = inertia

# ── 3. Programmatically locate the elbow using the KneeLocator ──────────────
kl = KneeLocator(
    list(K_range), inertia_values,
    curve='convex', direction='decreasing', S=1.0
)
optimal_k = kl.elbow
print(f"\n{'=' * 70}")
print(f"OPTIMAL K (Elbow Point): {optimal_k}")
print(f"{'=' * 70}")

# ── 4. Plot 1: Elbow Method / Inertia Curve ─────────────────────────────────
fig, axes = plt.subplots(1, 2, figsize=(16, 6))

# Left panel – Inertia curve with elbow annotation
ax1 = axes[0]
ax1.plot(list(K_range), inertia_values, 'b-o', linewidth=2, markersize=8,
         markerfacecolor='steelblue', markeredgecolor='navy', label='WCSS (Inertia)')
ax1.axvline(x=optimal_k, color='red', linestyle='--', linewidth=1.5,
            label=f'Optimal K = {optimal_k}')
ax1.scatter([optimal_k], [inertia_values[optimal_k - 1]],
            color='red', s=200, zorder=5, edgecolors='darkred', linewidths=2,
            label=f'Elbow Point (K={optimal_k})')
ax1.set_title('Elbow Method – Within-Cluster Sum of Squares (WCSS)',
              fontsize=13, fontweight='bold')
ax1.set_xlabel('Number of Clusters (K)', fontsize=12)
ax1.set_ylabel('WCSS / Inertia', fontsize=12)
ax1.set_xticks(list(K_range))
ax1.legend(fontsize=10)
ax1.grid(True, alpha=0.3)

# Right panel – Percentage reduction (rate of change)
ax2 = axes[1]
pct_reductions = [r['Pct_Reduction'] for r in results[1:]]  # skip K=1
ax2.bar(range(2, 16), pct_reductions, color='steelblue', edgecolor='navy', alpha=0.8)
ax2.axvline(x=optimal_k, color='red', linestyle='--', linewidth=1.5,
            label=f'Optimal K = {optimal_k}')
ax2.set_title('Percentage Reduction in WCSS per Additional Cluster',
              fontsize=13, fontweight='bold')
ax2.set_xlabel('Number of Clusters (K)', fontsize=12)
ax2.set_ylabel('% Reduction in WCSS', fontsize=12)
ax2.set_xticks(range(2, 16))
ax2.legend(fontsize=10)
ax2.grid(True, alpha=0.3, axis='y')

plt.tight_layout()
plt.savefig('/home/claude/elbow_method_plot.png', dpi=150, bbox_inches='tight')
plt.close()
print("\nPlot saved: elbow_method_plot.png")

# ── 5. Plot 2: Detailed inertia curve with second derivative ────────────────
fig2, ax3 = plt.subplots(figsize=(10, 6))

# Compute second differences (discrete second derivative)
first_diff  = np.diff(inertia_values)
second_diff = np.diff(first_diff)

ax3.plot(list(K_range), inertia_values, 'b-o', linewidth=2.5, markersize=9,
         markerfacecolor='steelblue', markeredgecolor='navy')

# Shade the elbow region
ax3.axvspan(optimal_k - 0.3, optimal_k + 0.3, alpha=0.2, color='red',
            label=f'Elbow region (K={optimal_k})')
ax3.axvline(x=optimal_k, color='red', linestyle='--', linewidth=2)

# Annotate key points
for k_val in [1, optimal_k, 15]:
    idx = k_val - 1
    ax3.annotate(f'K={k_val}\nWCSS={inertia_values[idx]:,.0f}',
                 xy=(k_val, inertia_values[idx]),
                 xytext=(k_val + 0.8, inertia_values[idx] + 2000),
                 fontsize=9, fontweight='bold',
                 arrowprops=dict(arrowstyle='->', color='black'),
                 bbox=dict(boxstyle='round,pad=0.3', facecolor='lightyellow',
                           edgecolor='gray'))

ax3.set_title('Inertia Curve – K-Means Clustering\n'
              'Customer Churn Telecommunications Dataset',
              fontsize=14, fontweight='bold')
ax3.set_xlabel('Number of Clusters (K)', fontsize=12)
ax3.set_ylabel('Within-Cluster Sum of Squares (WCSS)', fontsize=12)
ax3.set_xticks(list(K_range))
ax3.legend(fontsize=11)
ax3.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('/home/claude/inertia_curve_detailed.png', dpi=150, bbox_inches='tight')
plt.close()
print("Plot saved: inertia_curve_detailed.png")

# ── 6. Save tabular results ─────────────────────────────────────────────────
results_df = pd.DataFrame(results)
results_df.to_csv('/home/claude/elbow_method_results.csv', index=False)
print("Results saved: elbow_method_results.csv")

# ── 7. Summary ───────────────────────────────────────────────────────────────
print(f"\n{'=' * 70}")
print("SUMMARY")
print(f"{'=' * 70}")
print(f"Optimal number of clusters (K): {optimal_k}")
print(f"WCSS at optimal K:              {inertia_values[optimal_k - 1]:,.2f}")
print(f"WCSS at K=1 (baseline):         {inertia_values[0]:,.2f}")
print(f"Total variance explained by "
      f"K={optimal_k}: {(1 - inertia_values[optimal_k-1]/inertia_values[0])*100:.1f}%")
print(f"{'=' * 70}")
