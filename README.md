# 🏦 Credit Risk Assessment using Linear Algebra Techniques

> *"We model credit risk using distance from an ideal borrower, reduce feature noise using PCA, and analyze feature importance using eigenvalues."*

---

## 📌 Project Summary

This project builds a mathematically rigorous **Credit Risk Scoring System** using core Linear Algebra techniques. Given a dataset of 20 loan applicants described by 6 financial features, the system computes a composite risk score for each customer, ranks them from safest to riskiest, and visualizes the underlying structure of the data using Principal Component Analysis (PCA).

The approach is entirely interpretable — no black-box ML models. Every score is derived from vector operations, matrix decompositions, and linear equations covered in a standard Linear Algebra syllabus.

---

## 👨‍💻 Team Details

| Field | Details |
|---|---|
| **Subject** | Linear Algebra / Applied Mathematics |
| **Topic** | Credit Risk Assessment using Linear Algebra |
| **Tools Used** | Python, NumPy, Matplotlib, Seaborn |

---

## 🌍 Real-World Problem Statement

Banks and financial institutions face the challenge of evaluating loan applicants fairly and accurately.

- Many factors are involved: income, expenses, credit score, loan amount, employment history, and existing debt
- Data can be noisy or correlated across features
- A mathematical, objective scoring system is needed to reduce human bias

**Our Solution:** Use Linear Algebra techniques to build a Credit Risk Scoring System that:

1. Computes a risk score for each customer
2. Ranks customers from safest to riskiest
3. Uses PCA to reduce noise and extract key features
4. Uses Eigenvalues to identify the most important risk factors

---

## 🧩 Syllabus Coverage

| Unit | Concept Used | Application |
|---|---|---|
| Unit 1 | System of Linear Equations (`np.linalg.lstsq`) | Weighted Score Calculation via Least Squares |
| Unit 2 | Vector Norms (`np.linalg.norm`) | Distance from Ideal Borrower |
| Unit 3 | Eigenvalues & Eigenvectors | Feature Importance Analysis |
| Unit 3 | PCA (Principal Component Analysis) | Dimensionality Reduction / Noise Removal |

---

## 📁 Dataset

The dataset consists of **20 synthetic loan applicants**, each described by **6 financial features**:

| Feature | Description |
|---|---|
| `Income` | Annual gross income (USD) |
| `Expenses` | Annual expenses (USD) |
| `Credit Score` | FICO-style score (300–850) |
| `Loan Amount` | Requested loan amount (USD) |
| `Emp. Years` | Years of continuous employment |
| `Existing Debt` | Outstanding debt balance (USD) |

---

## ⚙️ Methodology

### Step 1 — Z-Score Normalization
All features are normalized to zero mean and unit variance so that features with larger numerical ranges (e.g., Income in USD) do not dominate features with smaller ranges (e.g., Employment Years).

```
normalized = (X - mean) / std
```

### Step 2 — Polarity Alignment
Features are assigned a polarity vector `[+1, -1, +1, -1, +1, -1]` to ensure that "higher is better" for positive features (Income, Credit Score, Employment Years) and "lower is better" for negative ones (Expenses, Loan Amount, Existing Debt). The normalized data is multiplied element-wise by this vector.

### Step 3 — Least Squares Weighting (Unit 1)
A target vector `b` is constructed from derived financial ratios (debt-to-income, loan-to-income, credit quality). The system solves:

```
A · w ≈ b    →    w = lstsq(A, b)
```

This yields a weight vector `w` that captures how much each feature contributes to creditworthiness. The least-squares solution minimizes the residual error across all 20 customers simultaneously.

### Step 4 — Distance from Ideal Borrower (Unit 2)
An "ideal profile" is constructed as the best observed value across all customers for each feature. The Euclidean distance of each customer from this ideal is computed:

```
risk_distance[i] = ||aligned[i] - ideal||₂
```

A larger distance = further from the ideal = higher risk.

### Step 5 — Final Composite Score
The two risk signals (distance score and least-squares score) are min-max normalized and combined with a weighted average:

```
final_score = minmax(0.6 × norm_distance − 0.4 × norm_ls_score)
```

Customers are then bucketed into **Low Risk**, **Medium Risk**, and **High Risk** using 33rd and 66th percentile thresholds.

### Step 6 — PCA (Unit 3)
The covariance matrix of the aligned data is computed, and its eigenvalues and eigenvectors are extracted. The top two principal components are used to project the 6-dimensional data into 2D for visualization. The PC1 loading bar chart reveals which original features drive the most variance in the dataset.

---

## 📊 Outputs

The project produces the following outputs:

**Console Output**
- Z-score normalized data (first 3 rows)
- Least squares weights per feature
- Ideal borrower profile
- Full risk ranking table (safest → riskiest)
- PCA variance explained per component
- Summary table with Income, Credit Score, LS Score, Distance, Final Score, and Label
- Risk label distribution (Low / Medium / High)

**Visualization (3-panel figure)**
- **Bar Chart** — Risk scores for all 20 customers, color-coded by risk label
- **PCA Scatter Plot** — 2D projection of customers, colored by risk category
- **PC1 Loadings Bar Chart** — Feature contributions to the first principal component

---

## 🚀 How to Run

### Prerequisites

```bash
pip install numpy matplotlib seaborn
```

### Execution

```bash
python credit_risk_assessment.py
```

No external data files are needed — the dataset is embedded directly in the script.

---

## 📦 File Structure

```
credit_risk_assessment/
│
├── credit_risk_assessment.py   # Main project script (all steps in one file)
└── README.md                   # This file
```

---

## 🔍 Sample Output (Risk Ranking)

```
🏆 Risk Ranking (Safest → Riskiest):
Rank  Customer   Score    Label
-------------------------------------
1     C13        0.000    Low Risk
2     C4         0.052    Low Risk
3     C1         0.118    Low Risk
...
19    C6         0.934    High Risk
20    C12        1.000    High Risk
```

---

## 💡 Key Insights

- **High-income, low-debt customers** with strong credit scores consistently land in the Low Risk category, validating the model's financial logic.
- **PC1** captures the largest share of variance and loads heavily on Income (+) and Existing Debt (−), confirming these as the dominant risk dimensions.
- The **least squares solution** effectively learns implicit weights from derived financial ratios, acting as a data-driven alternative to manually tuned scorecards.
- **PCA clustering** shows clear separation between risk groups in 2D, suggesting the 6 features encode sufficient signal for discrimination without any machine learning.

---
## Screenshots of output

<img width="1600" height="859" alt="WhatsApp Image 2026-04-21 at 16 29 31" src="https://github.com/user-attachments/assets/5026bf8c-f380-448c-b164-9583fdf90650" />


---

## 📚 References

- Strang, G. (2016). *Introduction to Linear Algebra* (5th ed.). Wellesley-Cambridge Press.
- NumPy Documentation — `np.linalg.lstsq`, `np.linalg.eig`, `np.linalg.norm`
- Jolliffe, I. T. (2002). *Principal Component Analysis* (2nd ed.). Springer.
