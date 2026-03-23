# 📊 Unsupervised Anomaly Detection in Stock Market Data using PCA & DBSCAN

## 🚀 Overview

This project implements an unsupervised machine learning pipeline to detect anomalies in stock market data. Using dimensionality reduction (PCA) and density-based clustering (DBSCAN), the model identifies unusual patterns in price and volume behavior without requiring labeled data.

---

## 📁 Dataset

The dataset contains the following features:

* Open
* High
* Low
* Close
* Adjusted Close
* Volume

These features represent stock price movements and trading activity over time.

---

## 🧠 Approach

### 1. Data Preprocessing

* Removed non-numeric column (`Date`)
* Standardized features using `StandardScaler` to ensure equal contribution

### 2. Dimensionality Reduction (PCA)

* Reduced dataset to 2 principal components
* Explained variance:

  * PC1 ≈ 83.45%
  * PC2 ≈ 16.45%
* Total variance retained ≈ 99.9%

**Interpretation:**

* PC1 → Overall price movement
* PC2 → Trading volume influence

---

### 3. Clustering using DBSCAN

* Algorithm: Density-Based Spatial Clustering
* Parameters:

  * `eps = 0.335` (determined using k-distance elbow method)
  * `min_samples = 7`

**Why DBSCAN?**

* Does not require number of clusters beforehand
* Naturally detects outliers (anomalies)
* Works well with arbitrarily shaped clusters

---

### 4. Anomaly Detection

* Points labeled `-1` by DBSCAN are considered anomalies
* These represent unusual price-volume relationships

---

## 📈 Results

* Total Data Points: 503
* Number of Anomalies: 95
* Percentage of Anomalies: 18.89%

### 🔍 Cluster Evaluation

* Silhouette Score: 0.20825505750055617

---

## 📊 Visualization

* X-axis → PC1 (Overall Price Movement)
* Y-axis → PC2 (Volume Influence)
* Clusters are color-coded
* Anomalies are highlighted in **red**

---

## 💡 Insights

* The model effectively separates normal trading patterns from unusual activity
* Anomalies may indicate:

  * sudden market spikes
  * abnormal trading volumes
  * potential market irregularities

---

## 🛠️ Tech Stack

* Python
* Pandas
* Scikit-learn
* Matplotlib

---

## 🔮 Future Improvements

* Compare with other anomaly detection methods (Isolation Forest, LOF)
* Apply the same pipeline to different datasets (e.g., air quality data)
* Automate parameter tuning for DBSCAN

---

## 📌 Conclusion

This project demonstrates how unsupervised learning techniques can be used to extract meaningful insights and detect anomalies in financial data without labeled outputs.

---
