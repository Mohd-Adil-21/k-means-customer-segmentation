# Customer Segmentation via K-Means Clustering

An unsupervised machine learning project using Scikit-Learn to partition unlabelled customer behavioral and financial data into distinct target demographics using K-Means clustering.

---

## 📌 Project Overview
Customer segmentation enables organizations to execute tailored marketing campaigns and allocate promotional budgets effectively. This project applies unsupervised clustering to an unlabelled dataset (Mall Customers) to:
* Determine the optimal number of clusters using mathematical heuristic criteria (Elbow Method & Silhouette Coefficient).
* Normalize multivariable continuous data to mitigate distance distortion in Euclidean space.
* Segment consumer groups based on **Annual Income** and **Spending Score**.
* Map mathematical clusters to actionable commercial consumer personas.

---

## ⚙️ Methodology & Architecture
1. **Data Preprocessing & Scaling:**
   * Handled tabular inputs and scaled features using `StandardScaler` to ensure unbiased Euclidean distance measurements.
2. **Cluster Optimization:**
   * **Elbow Method:** Evaluated Within-Cluster Sum of Squares (Inertia) across $K \in [2, 10]$ to isolate the point of diminishing marginal returns.
   * **Silhouette Analysis:** Calculated separation-to-cohesion ratios, confirming the strongest geometric partitioning at $K = 5$.
3. **Model Fitting:**
   * Executed `KMeans` with `init='k-means++'` to guarantee stable centroid initialization and reproducible convergence.
4. **Behavioral Profiling:**
   * Extracted statistical centroids to label distinct cohorts: *High-Income/High-Spenders*, *Savers*, *Balanced*, *Careless Spenders*, and *Budget Shoppers*.

---

## 📊 Evaluation & Empirical Metrics
* **Optimal Clusters ($N$):** 5 distinct groups
* **Within-Cluster Variance (Inertia):** Reduced from $>250.0$ to $65.57$ at the selected inflection point
* **Silhouette Score:** 0.554 (indicating strong cluster cohesion and separation)

---

## 🛠️ Tech Stack
* **Language:** Python
* **Machine Learning:** Scikit-learn
* **Data Processing & Math:** Pandas, NumPy
* **Data Visualization:** Matplotlib, Seaborn
* **Environment:** Google Colab / Jupyter Notebook

---

## 🚀 How to Run Locally

1. Clone the repository:
```bash
git clone [https://github.com/Mohd-Adil-21/kmeans-customer-segmentation.git](https://github.com/Mohd-Adil-21/kmeans-customer-segmentation.git)
cd kmeans-customer-segmentation
pip install pandas numpy scikit-learn matplotlib seaborn
python kmeans_customer_segmentation.py
