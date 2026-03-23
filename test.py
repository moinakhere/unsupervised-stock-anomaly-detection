from sklearn.cluster import DBSCAN
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score
import matplotlib.pyplot as plt

# Data fetch
datas = pd.read_csv("data.csv")
dataset = pd.DataFrame(datas).drop(["Date"], axis=1)

# Scaling Data
scaler=StandardScaler()
scaled_data=scaler.fit_transform(dataset)
scaled_dataset=pd.DataFrame(
    scaled_data,
    columns=dataset.columns
    )

# Applying PCA to reduce the dataset into 2 dimensions
pca = PCA(n_components=2)  # 2 is chosen as it compiles almost 99.9% of the whole dataset, namely PC1=0.83458871 and PC2=0.16456432
pca_data = pca.fit_transform(scaled_dataset)

# Applying DBSCAN for clustering as the data is in form of globules
dbscan = DBSCAN(eps=0.335, min_samples=7) #0.335 is chosen since the elbow point comes between 0.32 and0.35
labels = dbscan.fit_predict(pca_data)

# Number of anomalies calculated, which can be a rare event or noise
n_anomalies = list(labels).count(-1)
print("Number of clusters:", len(set(labels)) - (1 if -1 in labels else 0))
print(f"Number of anomalies detected: {n_anomalies}")
print(f"Total data points: {len(labels)}")
print(f"Percentage of anomalies: {n_anomalies / len(labels) * 100:.2f}%")

# Anomalies and clusters are seperated
anomalies=pca_data[labels == -1]
clusters = pca_data[labels != -1]
cleaned_labels = labels[labels != -1]

# Silhouette Score
print("Silhouette Score:", silhouette_score(clusters, cleaned_labels))

# Visualization
plt.figure(figsize=(10,8),dpi=100)
plt.scatter(clusters[:, 0], clusters[:, 1], c=cleaned_labels, cmap='viridis')
plt.scatter(anomalies[:, 0], anomalies[:, 1], color='red')
plt.title("Clustering with Anomalies of Stock Price Movement")
plt.xlabel("PC1(Overall Price Movement)")
plt.ylabel("PC2(Trading Volume)")
plt.show()
