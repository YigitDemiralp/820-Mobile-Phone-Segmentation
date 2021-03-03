import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import missingno as msno
from datetime import datetime, date
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn import metrics 
import scikitplot as skplt

df_train = pd.read_csv(r'C:\Users\YEET\Documents\GitHub\820-Mobile-Phone-Segmentation\train.csv')
df_test = pd.read_csv(r'C:\Users\YEET\Documents\GitHub\820-Mobile-Phone-Segmentation\test.csv')

df_train.shape
df_train

scaler = StandardScaler()
scaled_df = scaler.fit_transform(df_train)

pca = PCA()
pcs = pca.fit_transform(scaled_df)
varexp = pca.explained_variance_ratio_
sns.lineplot(range(1, len(varexp)+1), varexp)
plt.title('Explained Variance')
plt.xlabel('Number of Components')
plt.ylabel('Variance')
plt.show()


plt.title("Cumulative Explained Variance")
plt.plot(range(1, len(varexp)+1), np.cumsum(varexp))
plt.axhline(.95, color = 'r')
plt.show()


KRANGE = range(2, 21)
inertias = []
for k in ks:
    # Create a KMeans instance with k clusters: model
    model = KMeans(n_clusters = k,  random_state = 455)
    
    # Fit model to samples
    model.fit(scaled_df)
    
    # Append the inertia to the list of inertias
    inertias.append(model.inertia_)
    
# Plot ks vs inertias
plt.plot(ks, inertias, '-o')
plt.xlabel('number of clusters, k')
plt.ylabel('inertia')
plt.xticks(ks)
plt.title('Selecting Number of Clusters')
plt.show()



# containers
ss = []

for k in KRANGE:
 km = KMeans(k)
 lab = km.fit_predict(scaled_df)
 ss.append(metrics.silhouette_score(scaled_df, lab))
# the plot 
sns.lineplot(KRANGE, ss)
plt.xticks(ks)
plt.show()


k6 = KMeans(6)
k6_labs = k6.fit_predict(scaled_df)

# plot
skplt.metrics.plot_silhouette(scaled_df, k6_labs)
plt.show()