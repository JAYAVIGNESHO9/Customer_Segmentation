
# CUSTOMER SEGMENTATION USING K-MEANS CLUSTERING


import numpy as np
import pandas as pd

import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

import warnings
warnings.filterwarnings("ignore")



df = pd.read_csv("Mall_Customers.csv")

print("Dataset Loaded Successfully!\n")

# Display first rows
print(df.head())

print("\nDataset Info:\n")
print(df.info())

print("\nMissing Values:\n")
print(df.isnull().sum())

# Remove missing values if any
df = df.dropna()

# Convert Gender to numeric
df['Gender'] = df['Gender'].map({
    'Male': 0,
    'Female': 1
})

# Drop CustomerID
if 'CustomerID' in df.columns:
    df = df.drop('CustomerID', axis=1)

print("\nCleaned Dataset Preview:\n")
print(df.head())

X = df[['Age',
        'Annual Income (k$)',
        'Spending Score (1-100)']]

scaler = StandardScaler()

X_scaled = scaler.fit_transform(X)

wcss = []

for i in range(1, 11):

    kmeans = KMeans(
        n_clusters=i,
        init='k-means++',
        random_state=42
    )

    kmeans.fit(X_scaled)

    wcss.append(kmeans.inertia_)



# Plot Elbow Curve

plt.figure(figsize=(8,5))

plt.plot(range(1,11), wcss, marker='o')

plt.title("Elbow Method for Optimal K")

plt.xlabel("Number of Clusters")

plt.ylabel("WCSS")

plt.show()


# Choose K (usually 5 from elbow)
kmeans = KMeans(
    n_clusters=5,
    init='k-means++',
    random_state=42
)

clusters = kmeans.fit_predict(X_scaled)

# Add cluster labels
df['Cluster'] = clusters

print("\nClusters Assigned Successfully!")

plt.figure(figsize=(8,6))

sns.scatterplot(
    x='Annual Income (k$)',
    y='Spending Score (1-100)',
    hue='Cluster',
    data=df,
    palette='Set1',
    s=100
)

plt.title("Customer Segments Visualization")

plt.show()

cluster_profile = df.groupby('Cluster').mean()

print("\nCluster Profile:\n")
print(cluster_profile)


cluster_counts = df['Cluster'].value_counts()

print("\nNumber of Customers per Cluster:\n")
print(cluster_counts)

centers = scaler.inverse_transform(
    kmeans.cluster_centers_
)

centers_df = pd.DataFrame(
    centers,
    columns=[
        'Age',
        'Annual Income (k$)',
        'Spending Score (1-100)'
    ]
)

print("\nCluster Centers:\n")
print(centers_df)

output_file = "Customer_Segments_Output.csv"

df.to_csv(output_file, index=False)

print("\nOutput saved as:", output_file)

print("\n==============================")
print("SEGMENT REPORTS")
print("==============================")

for cluster in sorted(df['Cluster'].unique()):

    print("\n----------------------------------")
    print("Cluster", cluster)
    print("----------------------------------")

    subset = df[df['Cluster'] == cluster]

    avg_age = round(
        subset['Age'].mean(), 2)

    avg_income = round(
        subset['Annual Income (k$)'].mean(), 2)

    avg_spending = round(
        subset['Spending Score (1-100)'].mean(), 2)

    print("Average Age:", avg_age)
    print("Average Income:", avg_income)
    print("Average Spending Score:", avg_spending)

    

    # Marketing Suggestions

    if avg_income > 70 and avg_spending > 70:

        print("Marketing Action:")
        print("Target with VIP services and premium offers.")

    elif avg_spending > 70:

        print("Marketing Action:")
        print("Promote new products and loyalty rewards.")

    elif avg_income < 40 and avg_spending < 40:

        print("Marketing Action:")
        print("Offer discounts and budget deals.")

    else:

        print("Marketing Action:")
        print("Use personalized promotions.")


from mpl_toolkits.mplot3d import Axes3D

fig = plt.figure(figsize=(8,6))

ax = fig.add_subplot(111, projection='3d')

ax.scatter(
    df['Age'],
    df['Annual Income (k$)'],
    df['Spending Score (1-100)'],
    c=df['Cluster']
)

ax.set_xlabel("Age")
ax.set_ylabel("Income")
ax.set_zlabel("Spending")

plt.title("3D Customer Segmentation")

plt.show()





print("\nCustomer Segmentation Project Completed Successfully!")