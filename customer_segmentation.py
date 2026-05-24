# import pandas as pd
# import matplotlib.pyplot as plt

# from sklearn.cluster import KMeans
# from sklearn.preprocessing import StandardScaler

# # Load dataset
# df = pd.read_csv("dataset/Mall_Customers.csv")

# print(df.head())

# # Select features
# X = df[['Annual Income (k$)', 'Spending Score (1-100)']]

# # Scaling
# scaler = StandardScaler()
# X_scaled = scaler.fit_transform(X)

# # KMeans model
# kmeans = KMeans(n_clusters=5, random_state=42)

# # Predict clusters
# df['Cluster'] = kmeans.fit_predict(X_scaled)

# print(df.head())
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

# Load dataset
df = pd.read_csv("dataset/Mall_Customers.csv")

# Show dataset
print(df.head())

# Select columns
X = df[['Annual Income (k$)', 'Spending Score (1-100)']]

# Scale data
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Elbow Method
wcss = []

for i in range(1,11):
    kmeans = KMeans(n_clusters=i, random_state=42)
    kmeans.fit(X_scaled)
    wcss.append(kmeans.inertia_)

# Plot elbow graph
plt.plot(range(1,11), wcss, marker='o')
plt.title("Elbow Method")
plt.xlabel("Clusters")
plt.ylabel("WCSS")
plt.show()

# Apply KMeans
kmeans = KMeans(n_clusters=5, random_state=42)

# Predict clusters
df['Cluster'] = kmeans.fit_predict(X_scaled)

# Visualize clusters
plt.scatter(
    X_scaled[:,0],
    X_scaled[:,1],
    c=df['Cluster']
)

plt.xlabel("Annual Income")
plt.ylabel("Spending Score")
plt.title("Customer Segments")
plt.show()

print(df.head())
   

# Apply KMeans
kmeans = KMeans(n_clusters=5, random_state=42)

# Create cluster column
df['Cluster'] = kmeans.fit_predict(X_scaled)

# Save updated dataset
df.to_csv("dataset/customer_segmentation.csv", index=False)

print("File saved")