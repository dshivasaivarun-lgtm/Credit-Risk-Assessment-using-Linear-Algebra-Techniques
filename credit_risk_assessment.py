import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sns
from collections import Counter

plt.rcParams['figure.figsize'] = (15, 11)
sns.set_theme(style="whitegrid")
np.random.seed(42)

print("Libraries imported successfully!")

# ---------------------------------------------------------------
# Step 1: Dataset
# ---------------------------------------------------------------
data = np.array([
    [80000, 18000, 760, 120000,  8,  5000],
    [55000, 26000, 680, 200000,  4, 25000],
    [42000, 35000, 590, 310000,  2, 60000],
    [95000, 12000, 800,  80000, 12,  2000],
    [60000, 22000, 720, 170000,  5, 15000],
    [38000, 40000, 560, 350000,  1, 75000],
    [72000, 19000, 740, 140000,  7,  8000],
    [48000, 30000, 630, 260000,  3, 40000],
    [88000, 14000, 780, 100000, 10,  3500],
    [52000, 28000, 660, 220000,  4, 30000],
    [65000, 21000, 730, 160000,  6, 12000],
    [35000, 42000, 540, 380000,  1, 85000],
    [100000,10000, 820,  60000, 15,  1000],
    [44000, 33000, 610, 290000,  2, 55000],
    [76000, 17000, 755, 130000,  9,  6500],
    [58000, 24000, 695, 190000,  5, 20000],
    [40000, 38000, 575, 330000,  1, 70000],
    [83000, 15000, 770, 110000, 11,  4000],
    [50000, 29000, 645, 240000,  3, 35000],
    [69000, 20000, 735, 150000,  7, 10000],
], dtype=float)

feature_names = ['Income', 'Expenses', 'Credit Score', 'Loan Amount', 'Emp. Years', 'Existing Debt']
n_customers, n_features = data.shape

print(f"\n Dataset: {n_customers} customers, {n_features} features")

# ---------------------------------------------------------------
# Step 2: Normalization
# ---------------------------------------------------------------
mean_vec = np.mean(data, axis=0)
std_vec  = np.std(data, axis=0)
normalized = (data - mean_vec) / std_vec

print("\nZ-Score Normalized Data (first 3 rows):")
for i in range(3):
    print(f"  Customer {i+1}: {np.round(normalized[i], 3)}")
print("  Column means:", np.round(np.mean(normalized, axis=0), 5))

# ---------------------------------------------------------------
# Step 3: Polarity
# ---------------------------------------------------------------
polarity = np.array([+1, -1, +1, -1, +1, -1])
aligned  = normalized * polarity

print("\n Polarity vector:", polarity)
print("   Features:", feature_names)

# ---------------------------------------------------------------
# Step 4: Least Squares
# ---------------------------------------------------------------
debt_to_income = data[:, 5] / data[:, 0]
loan_to_income = data[:, 3] / data[:, 0]
credit_scaled  = (data[:, 2] - 300) / 550

b = (credit_scaled - 0.5 * debt_to_income - 0.3 * loan_to_income).reshape(-1, 1)

weights, residuals, rank, sv = np.linalg.lstsq(aligned, b, rcond=None)

print("\n Least Squares Weights:")
for fname, w in zip(feature_names, weights.flatten()):
    print(f"   {fname:15s}: {w:+.4f}")
print(f"   Matrix rank: {rank}/{n_features}")

ls_scores = (aligned @ weights).flatten()

# ---------------------------------------------------------------
# Step 5: Distance
# ---------------------------------------------------------------
ideal = np.max(aligned, axis=0)
risk_distance = np.linalg.norm(aligned - ideal, axis=1)

print("\n Ideal Profile:")
for fname, v in zip(feature_names, ideal):
    print(f"   {fname:15s}: {v:.3f}")

# ---------------------------------------------------------------
# Step 6: Final Score
# ---------------------------------------------------------------
def minmax(x):
    return (x - x.min()) / (x.max() - x.min())

norm_dist = minmax(risk_distance)
norm_ls   = minmax(ls_scores)

final_score = minmax(0.6 * norm_dist - 0.4 * norm_ls)

low_t  = np.percentile(final_score, 33)
high_t = np.percentile(final_score, 66)

labels = []
for s in final_score:
    if s < low_t:
        labels.append('Low Risk')
    elif s < high_t:
        labels.append('Medium Risk')
    else:
        labels.append('High Risk')

ranking = np.argsort(final_score)

print("\n🏆 Risk Ranking (Safest → Riskiest):")
print(f"{'Rank':<5} {'Customer':<10} {'Score':<8} {'Label'}")
print('-'*38)
for i, idx in enumerate(ranking):
    print(f"{i+1:<5} C{idx+1:<9} {final_score[idx]:<8.3f} {labels[idx]}")

# ---------------------------------------------------------------
# Step 7: PCA
# ---------------------------------------------------------------
A_centered = aligned - np.mean(aligned, axis=0)
cov_matrix = np.cov(A_centered.T)

eigenvalues, eigenvectors = np.linalg.eig(cov_matrix)
eigenvalues = np.real(eigenvalues)
eigenvectors = np.real(eigenvectors)

idx = np.argsort(eigenvalues)[::-1]
eigenvalues = eigenvalues[idx]
eigenvectors = eigenvectors[:, idx]

variance = eigenvalues / np.sum(eigenvalues)
cumulative = np.cumsum(variance)

print("\n Principal Components:")
for i in range(n_features):
    print(f"PC{i+1}: var={variance[i]*100:.1f}%  cumulative={cumulative[i]*100:.1f}%")

projected = A_centered @ eigenvectors[:, :2]

# ---------------------------------------------------------------
# Step 8: Visualization (clean layout)
# ---------------------------------------------------------------
cmap = {'Low Risk':'green','Medium Risk':'orange','High Risk':'red'}
colors = [cmap[l] for l in labels]

fig = plt.figure()
gs = gridspec.GridSpec(2,2)

# Bar chart
ax1 = fig.add_subplot(gs[0, :])
ax1.bar(range(1,n_customers+1), final_score, color=colors)
ax1.set_title("Risk Scores")

# PCA
ax2 = fig.add_subplot(gs[1,0])
for label in cmap:
    idx_l = [i for i,l in enumerate(labels) if l==label]
    ax2.scatter(projected[idx_l,0], projected[idx_l,1], c=cmap[label], label=label)
ax2.set_title("PCA Projection")
ax2.legend()

# Loadings
ax3 = fig.add_subplot(gs[1,1])
pc1 = eigenvectors[:,0]
ax3.barh(feature_names, pc1)
ax3.set_title("PC1 Loadings")

plt.tight_layout()
plt.show()

# ---------------------------------------------------------------
# Summary Table
# ---------------------------------------------------------------
print('\n' + '='*72)
print(f"{'Cust':<6}{'Income':>10}{'Credit':>8}{'LS Score':>10}{'Dist':>10}{'Final':>8}  Label")
print('-'*72)
for i in range(n_customers):
    print(f"C{i+1:<5}{data[i,0]:>10,.0f}{data[i,2]:>8.0f}"
          f"{ls_scores[i]:>10.3f}{risk_distance[i]:>10.3f}"
          f"{final_score[i]:>8.3f}  {labels[i]}")
print('='*72)

# Distribution
dist = Counter(labels)
print("\nDistribution:")
for k in dist:
    print(f"{k}: {dist[k]}")

print("\n Project Completed Successfully!")