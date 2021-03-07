#Hierarchical Clustering
# import basics packcages
import pandas as pd
import numpy as np
from mlxtend.frequent_patterns import apriori, association_rules
import seaborn as sns
import matplotlib.pyplot as plt
import plotly.express as px

# for distance and h-clustering
from scipy.cluster.hierarchy import linkage, dendrogram, fcluster
from scipy.spatial.distance import pdist, squareform

#Kmeans clustering
from sklearn.cluster import KMeans, DBSCAN
from sklearn.neighbors import NearestNeighbors
from sklearn import metrics 
from sklearn.decomposition import PCA
import scikitplot as skplt

# sklearn does have some functionality too, but mostly a wrapper to scipy
from sklearn.metrics import pairwise_distances 
from sklearn.preprocessing import StandardScaler

mobile = pd.read_csv("train.csv")
mobile.shape

#Distance for numeric
cols = ['blue','dual_sim','four_g','n_cores','three_g','touch_screen','wifi','price_range']

for col in cols:
    mobile[col] = mobile[col].astype('category')

mobile.info()

mobile_numeric = mobile.select_dtypes('number')
mobile_cat = mobile.select_dtypes('category')


#1) Normalize the data
sc = StandardScaler()
mobile_scaled = sc.fit_transform(mobile_numeric)
mobile_scaled = pd.DataFrame(mobile_scaled, columns = mobile_numeric.columns)


#2) Clustering using euclidean and cosine for distance matrix
dc1 = pdist(mobile_scaled.values) #euclidean
dc2 = pdist(mobile_scaled.values, metric='cosine')


#3) jaccard distance
mobile_ct = pd.get_dummies(mobile_cat, drop_first=True)
mobile_j = pdist(mobile_ct, metric='jaccard')

dist1 = dc1 + mobile_j
dist2 = dc2 + mobile_j

squareform(dist1)
squareform(dist2)

#See now with linkage methods and euclidean distance matrix work
METHODS = ['single', 'complete', 'average', 'ward']

plt.figure(figsize=(20,5))

for i,m in enumerate(METHODS):
    plt.subplot(1,4,i+1)
    plt.title(m)
    dendrogram(linkage(dist1, method=m),
                leaf_rotation= 90)

plt.show()



plt.figure(figsize=(20,5))
#See now with linkage methods and cosine distance matrix work
for i,m in enumerate(METHODS):
    plt.subplot(1,4,i+1)
    plt.title(m)
    dendrogram(linkage(dist2, method=m),
                leaf_rotation= 90)
plt.show()


#Distance euclidean
link_s = linkage(dist1, method='single') #bad
link_c = linkage(dist1, method='complete') #Good one, only one cluster with 768
link_a = linkage(dist1, method='average') #Average, we see more difference
link_w_1= linkage(dist1, method='ward') #Good one --> cluster 400's and one in 207 #favorite


plt.title('Dendogram for Euclidean and Ward')
dendrogram(link_w_1,
            leaf_rotation= 90)
plt.axhline(linestyle='--', y=32) #5 clusters
plt.axhline(linestyle='--', color='green', y=34) #6 clusters
plt.xlabel('Mobiles')
plt.ylabel('Distance')
plt.show()

#4) Create the labels
labels = fcluster(link_w_1, 5, criterion='maxclust')
np.unique(labels)

#put the labels into the clean dataset
mobile['cluster_5'] = labels

#Review the dataset with the labels
mobile.head(3)

#How many stocks per cluster
mobile.cluster_5.value_counts(dropna=False, sort=False)

#4) Create the labels
labels6 = fcluster(link_w_1, 6, criterion='maxclust')
np.unique(labels6)

#put the labels into the clean dataset
mobile['cluster_6'] = labels6
mobile.cluster_6.value_counts(dropna=False, sort=False)

cols = ['blue','dual_sim','four_g','n_cores','three_g','touch_screen','wifi','price_range']
for col in cols:
    mobile[col] = mobile[col].astype('int64')

mobile_cluster_mean = mobile.groupby('cluster_5').mean()
plt.figure(figsize=(8,8))
sns.barplot(x=mobile_cluster_mean.index, y="price_range", data=mobile_cluster_mean)
plt.show()

X = mobile_scaled.values
sns.scatterplot(X[:,0],X[:,3],hue=mobile.cluster_5, cmap="rainbow").set(title='Stock - Hierarchical Clustering')
plt.show()

###################### COSINE ####################################

#Distance cosine
link_s = linkage(dist2, method='single') #bad bad
link_c = linkage(dist2, method='complete') #577
link_a = linkage(dist2, method='average') # not good not good
link_w = linkage(dist2, method='ward') 

plt.title('Dendogram for Cosine and Ward')
dendrogram(link_w,
            leaf_rotation= 90)
plt.axhline(linestyle='--', y=10) #6 clusters
plt.axhline(linestyle='--', color='green', y=11) #5 clusters
plt.xlabel('Mobiles')
plt.ylabel('Distance')
plt.show()

#4) Create the labels
labels_1 = fcluster(link_w, 5, criterion='maxclust')
np.unique(labels)

#put the labels into the clean dataset
mobile['cosine_cluster_5'] = labels_1

#Review the dataset with the labels
mobile.head(3)

#How many stocks per cluster
mobile.cosine_cluster_5.value_counts(dropna=False, sort=False)
mobile_cluster_mean1 = mobile.groupby('cosine_cluster_5').mean()

plt.figure(figsize=(8,8))
sns.barplot(x=mobile_cluster_mean1.index, y="price_range", data=mobile_cluster_mean1)
plt.show()

#5) Create the labels
labels_6_1 = fcluster(link_w, 6, criterion='maxclust')
np.unique(labels)

#put the labels into the clean dataset
mobile['cosine_cluster_6'] = labels_6_1

#Review the dataset with the labels
mobile.head(3)

#How many stocks per cluster
mobile.cosine_cluster_5.value_counts(dropna=False, sort=False)

#5) Create the labels
labels_7 = fcluster(link_w, 7, criterion='maxclust')
np.unique(labels)

#put the labels into the clean dataset
mobile['cosine_cluster_7'] = labels_7

#Review the dataset with the labels
mobile.head(3)

#How many stocks per cluster
mobile.cosine_cluster_7.value_counts(dropna=False, sort=False)

mobile.info()
mobile.groupby('cluster_5')['battery_power'].count()
x1 = mobile.cosine_cluster_5.value_counts(dropna=False, sort=False)
x1 = pd.DataFrame(x1, index=x1.index)

x2 = mobile.cosine_cluster_6.value_counts(dropna=False, sort=False)
x2 = pd.DataFrame(x2, index=x2.index)

x3 = mobile.cluster_5.value_counts(dropna=False, sort=False)
x3 = pd.DataFrame(x3, index=x3.index)

x4 = mobile.cluster_6.value_counts(dropna=False, sort=False)
x4 = pd.DataFrame(x4, index=x4.index)

x8 = mobile.cosine_cluster_7.value_counts(dropna=False, sort=False)
x8 = pd.DataFrame(x8, index=x8.index)


frames = [x1,x2,x3,x4]
result = pd.concat(frames,axis=1)

result.fillna(value=0, inplace=True)
cols1 = ['cosine_cluster_5','cosine_cluster_6','cluster_5','cluster_6']
for col in cols1:
    result[col] = result[col].astype('int64')
result

g = sns.heatmap(result, annot=True, fmt='d', cmap='Blues')
g.set_xticklabels(g.get_xticklabels(), rotation=0)
plt.xlabel('Clusters')
plt.title('Number of mobiles in each cluster')
plt.show()



############K-means ##############

k_mobile = pd.read_csv('train.csv')
cols = ['blue','dual_sim','four_g','n_cores','three_g','touch_screen','wifi','price_range']

for col in cols:
    k_mobile[col] = k_mobile[col].astype('category')

mobile_numeric = k_mobile.select_dtypes('number')
sc= StandardScaler()
mobile_scaled = sc.fit_transform(mobile_numeric)
mobile_scaled = pd.DataFrame(mobile_scaled, columns = mobile_numeric.columns)

##K Means for 5
k5 = KMeans(5, random_state=888)
k5.fit(mobile_scaled)
k5_labs = k5.predict(mobile_scaled)

k_mobile['k5'] = k5_labs

k_mobile.groupby('k5').mean()
k5 = k_mobile.k5.value_counts(dropna=False,sort=False)

#kMeans for 6
k6 = KMeans(6, random_state=888)
k6.fit(mobile_scaled)
k6_labs = k6.predict(mobile_scaled)

k_mobile['k6'] = k6_labs

k6 = k_mobile.k6.value_counts(dropna=False,sort=False)

#kMeans for 7
k7 = KMeans(7, random_state=888)
k7.fit(mobile_scaled)
k7_labs = k7.predict(mobile_scaled)

k_mobile['k7'] = k7_labs

k7 = k_mobile.k7.value_counts(dropna=False,sort=False)


x5 = pd.DataFrame(k5, index=k5.index)
x5.index = [1,2,3,4,5]
x6 = pd.DataFrame(k6, index=k6.index)
x6.index = [1,2,3,4,5,6]
x7 = pd.DataFrame(k7, index=k7.index)
x7.index = [1,2,3,4,5,6,7]

cframes = [x1,x8,x5,x7]

clusters = pd.concat(cframes,axis=1)

clusters.fillna(value=0, inplace=True)
cols1 = ['cosine_cluster_5','cosine_cluster_7','k5','k7']
for col in cols1:
    clusters[col] = clusters[col].astype('int64')
clusters

g1 = sns.heatmap(clusters, annot=True, fmt='d', cmap='Blues')
g1.set_xticklabels(g1.get_xticklabels(), rotation=0)
plt.xlabel('Clusters')
plt.title('Number of mobiles in each cluster')
plt.show()


################# #K-MEANS Evaluation #####################

mobile_tr=pd.read_csv("train.csv")
# standardize in a one liner
ds = StandardScaler().fit_transform(mobile_tr)
# lets fit kmeans
k5 = KMeans(5, random_state=888)
k5_labs = k5.fit_predict(ds)
# Silhouette plot
skplt.metrics.plot_silhouette(ds, k5_labs)
plt.show()

k6 = KMeans(6, random_state=888)
k6_labs = k6.fit_predict(ds)
# Silhouette plot
skplt.metrics.plot_silhouette(ds, k6_labs)
plt.show()

k7 = KMeans(7, random_state=888)
k7_labs = k7.fit_predict(ds)
# Silhouette plot
skplt.metrics.plot_silhouette(ds, k7_labs)
plt.show()

k8 = KMeans(8, random_state=888)
k8_labs = k8.fit_predict(ds)
# Silhouette plot
skplt.metrics.plot_silhouette(ds, k8_labs)
plt.show()

KRANGE = range(2,20)
sse = []

## loop over and evaluate
for k in KRANGE:
  km = KMeans(k)
  #labs = km.fit_predict(mobile_scaled)
  labs = km.fit_predict(ds)
  sse.append(km.inertia_)

#Elbow Method
plt.plot(KRANGE, sse, '-o', color='black')
plt.xlabel('number of clusters, k')
plt.ylabel('inertia')
#plt.axhline(y=33363, xmin=4,xmax=7)
plt.xticks(KRANGE)
plt.title('Selecting Number of Clusters')
plt.show()
#It looks like 5 is good number of clusters

# Testing K
ss1 = []

for k in KRANGE:
  km = KMeans(k)
  #lab = km.fit_predict(mobile_scaled)
  lab = km.fit_predict(ds)
  ss1.append(metrics.silhouette_score(ds, lab))

sns.lineplot(KRANGE, ss1)
plt.xlabel('number of clusters, k')
plt.ylabel('silhouette score')
plt.axvline(linestyle='--', x=5, color='red') #5clusters
plt.axvline(linestyle='--', color='green', x=6) #6 clusters
plt.axvline(linestyle='--', color='yellow', x=7) #6 clusters
plt.xticks(KRANGE)
plt.title('Selecting Number of Clusters')
plt.show()

########Cluster profiling ###########

kmobile_numeric = k_mobile.select_dtypes('number')
kmobile_numeric.drop(columns='k6',inplace=True)
kmobile_numeric.drop(columns='k7',inplace=True)
#stock_numeric.drop(columns='quarter_year', inplace=True)

clus_profile = kmobile_numeric.groupby("k5").mean()
clus_profile.index = [1,2,3,4,5]
# we can also plot this as a heatmap, but we should normalize the data
scp = StandardScaler()
cluster_scaled = scp.fit_transform(clus_profile)
cluster_scaled = pd.DataFrame(cluster_scaled, index=clus_profile.index, columns=clus_profile.columns)
plt.figure(figsize=(9,9))
sns.heatmap(cluster_scaled, cmap="Greens")
plt.show()


kmobile_numeric1 = k_mobile.select_dtypes('number')
kmobile_numeric1.drop(columns='k5',inplace=True)
kmobile_numeric1.drop(columns='k6',inplace=True)
clus_profile1 = kmobile_numeric1.groupby("k7").mean()
# we can also plot this as a heatmap, but we should normalize the data
scp = StandardScaler()
cluster_scaled = scp.fit_transform(clus_profile1)
cluster_scaled = pd.DataFrame(cluster_scaled, index=clus_profile1.index, columns=clus_profile1.columns)
plt.figure(figsize=(9,9))
sns.heatmap(cluster_scaled, cmap="Reds")
plt.show()



################ TOKENIZING AMAZON REVIEWS ###################