import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import missingno as msno
from datetime import datetime, date
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

df_train = pd.read_csv(r'C:\Users\YEET\Documents\GitHub\820-Mobile-Phone-Segmentation\train.csv')
df_test = pd.read_csv(r'C:\Users\YEET\Documents\GitHub\820-Mobile-Phone-Segmentation\test.csv')

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