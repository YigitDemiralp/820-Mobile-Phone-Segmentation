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
from scipy.cluster.hierarchy import linkage, dendrogram, fcluster
from scipy.spatial.distance import pdist, squareform

df_train = pd.read_csv(r'C:\Users\YEET\Documents\GitHub\820-Mobile-Phone-Segmentation\train.csv')
df_test = pd.read_csv(r'C:\Users\YEET\Documents\GitHub\820-Mobile-Phone-Segmentation\test.csv')

df_train.shape
df_train

scaler = StandardScaler()
scaled_df = scaler.fit_transform(df_train)



#PCA
pca = PCA()
pcs = pca.fit_transform(scaled_df)
varexp = pca.explained_variance_ratio_
sns.lineplot(range(1, len(varexp)+1), varexp)
plt.title('Explained Variance')
plt.xlabel('Number of Components')
plt.ylabel('Variance')
plt.show()

#Cumulative Explained Variance PCA
plt.title("Cumulative Explained Variance")
plt.plot(range(1, len(varexp)+1), np.cumsum(varexp))
plt.axhline(.95, color = 'r', label = '.95')
trans = plt.get_xaxis_transform() # x in data untis, y in axes fraction
plt.annotate('0.95', xy=(22, .95), xycoords='data', annotation_clip=False)
plt.show()

#KMeans
KRANGE = range(2, 20)
inertias = []
for k in KRANGE:
    # Create a KMeans instance with k clusters: model
    model = KMeans(n_clusters = k,  random_state = 455, )
    
    # Fit model to samples
    model.fit(scaled_df)
    
    # Append the inertia to the list of inertias
    inertias.append(model.inertia_)
    
# Plot ks vs inertias
plt.plot(KRANGE, inertias, '-o')
plt.xlabel('number of clusters, k')
plt.ylabel('inertia')
plt.xticks(KRANGE)
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
plt.xticks(KRANGE)
plt.show()


k6 = KMeans(6)
k6_labs = k6.fit_predict(scaled_df)

# plot
skplt.metrics.plot_silhouette(scaled_df, k6_labs)
plt.show()

numeric_columns = ['pc', 'talk_time', 'm_dep', 'battery_power', 'clock_speed', 'fc', 'int_memory', 'mobile_wt', 'px_height', 'px_width', 'ram', 'sc_h', 'sc_w']
df_numeric = df_train[numeric_columns]

categorical_cols = []
for i in df_train.columns:
    if i not in numeric_columns:
        categorical_cols.append(i)
df_cat = df_train[categorical_cols]

scaled_df_num = StandardScaler().fit_transform(df_numeric)
#hierarchical clustering
dist_mat = pdist(scaled_df, metric = 'jaccard')
linkage_array = linkage(dist_mat)

# y = list(np.unique(hclust))
# i = .619047618999995
# while True:
#     hclust = fcluster(linkage_array, i, criterion= 'distance')
#     y = list(np.unique(hclust))
#     if 5 not in y:
#         print(i)  
#         break
#     i += .00000000000001
        
hclust = fcluster(linkage_array, .619047618999995, criterion= 'distance')
y = list(np.unique(hclust))
y
df_train['HClust_Labels'] = hclust

scaled_df = StandardScaler().fit_transform(df_numeric)
dist_mat_num = pdist(scaled_df)
numeric_sf = squareform(dist_mat)
categorical_sf = squareform(dist_mat)




linkage_array = linkage(dist_mat)
hclust = fcluster(linkage_array, 5, criterion= 'maxclust')
df_train['HClust_Labels'] = hclust

df_train.groupby('HClust_Labels').describe()






df_cat_cd = pd.get_dummies(df_cat, drop_first=True)
dist_mat_cat = pdist(scaled_df, metric = 'jaccard')

scaled_df = StandardScaler().fit_transform(df_numeric)
dist_mat_num = pdist(scaled_df, metric = 'cosine')

distance_matrix_sum = dist_mat_num + dist_mat_cat
linkage_array = linkage(distance_matrix_sum)
hclust = fcluster(linkage_array, 20, criterion= 'maxclust')
df_train['HClust_Labels'] = hclust
df_train.groupby('HClust_Labels').describe()





corr = df_train.corr()
mask = np.triu(np.ones_like(corr, dtype=bool))
f, ax = plt.subplots(figsize=(10, 5))
cmap = sns.diverging_palette(230, 20, as_cmap=True)
sns.heatmap(corr, mask=mask, cmap=cmap, vmax=.3, center=0,
            square=True, linewidths=.5,vmin = -1, vmax = 1, cbar_kws={"shrink": .5}).set(title='Correlation Matrix')
plt.show()


sns.stripplot(x="price_range", y="ram",  data=df_train, palette="Set1")
plt.title('Price Range vs Ram')
plt.show()