import pandas as pd
import numpy as np 
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.datasets import make_blobs
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler

# Set random seed for reproducibility
np.random.seed(42)


# Generate synthetic customer data
X, y_true = make_blobs(
    n_samples=1000, centers=4,
    cluster_std=1, random_state=42)
X = X * [10000, 10]  
'''# Scale features to represent annual spend $ and
purchase frequency'''


# Create a pandas DF
df = pd.DataFrame(X, columns=['Annual_Spend', 'Purchase_Frequency'])
df['True_Cluster'] = y_true
print("\nCustomer Dataset (first 5 rows):")
print(df.head())


# Display summary statistics
print("\nDataset Summary Statistics:")
print(df.describe())


# Visualize the dataset: Scater plot of annual spend cs purchase frequenct
plt.figure(figsize=(8, 6))
sns.scatterplot(data=df, x='Annual_Spend', y='Purchase_Frequency', hue='True_Cluster', palette='deep')
plt.title('Customer Data: Annual Spend vs Purchase Frequecnt (True Clusters)')
plt.xlabel('Annual Spend $')
plt.ylabel('Purchase_Frequency')
plt.savefig('Customer_scatter_true.png')
plt.show()


# Verify the dataset size
print(f"\n Total samples: {len(X)}")


# Scale the features (K-Means is sensitive to scale)
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)


# Apply K-Means clustering (finl model with k=4)
kmeans_final = KMeans(n_clusters=4, random_state=42, n_init=10)
cluster_labels = kmeans_final.fit_predict(X_scaled)


# Add predicted clusters to df
df['Predicted_Cluster'] = cluster_labels


# Calculate silhouette score
sil_score = silhouette_score(X_scaled, cluster_labels)
print(f"\nSilhouette Score (k=4): {sil_score:.2f}")


# Visualize clustering results
plt.figure(figsize=(8, 6))
sns.scatterplot(data=df, x='Annual_Spend', y='Purchase_Frequency', hue='Predicted_Cluster', palette='deep')
plt.title('K-Means Clustering Results')
plt.xlabel('Annual Spend $')
plt.ylabel('Purchase Frequency')
plt.savefig('customer_scatter_predicted.png')
plt.show()


# Visualize silhoutte scores for different k values
sil_scores = []
k_values = range(2, 7)
for k in k_values:
    kmeans_tmp = KMeans(n_clusters=k, random_state=42, n_init=10)
    labels = kmeans_tmp.fit_predict(X_scaled)
    sil_scores.append(silhouette_score(X_scaled, labels))


plt.figure(figsize=(6, 4))
plt.plot(k_values, sil_scores, marker='o')
plt.title('Silhouette Score vd Number of Clusters')
plt.xlabel('Number of Clusters (k)')
plt.ylabel('Silhouette Score')
plt.savefig('silhouette scores.png')
plt.show()


# Example: Predict cluster for a new customer
new_customer = scaler.transform(np.array([[50000, 20]]))
# Example $50,000 spend, 20 purchases
predicted_cluster = kmeans_final.predict(new_customer)
print(f"\nPredicted cluster for new Customer (50,000$, 20 purchases): {predicted_cluster[0]}")
