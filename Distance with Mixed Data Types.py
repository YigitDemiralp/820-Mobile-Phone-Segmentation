import numpy as np
import pandas as pd
from scipy.spatial.distance import pdist

#Settings
Project = 'ba810-303000'

#quick tests from Rstudio
np.random.rand(5,4)


#goal of the lesson
# distance/clustering with non-numeric data types

#get the same cars dataset
cars = pd.read_gbq('SELECT * FROM `questrom.datasets.mtcars`',Project)

COLS = ['mpg','disp','hp','drat','wt','qsec']
cars2 = cars.copy()
cars2.index = cars2.model
del cars2['model']
cars_n = cars2.loc[:,COLS]
cars_c = cars2.drop(columns=COLS)

#distance mtrix for the numeric columns

#scale
from sklearn.preprocessing import StandardScaler

sc = StandardScaler()
cars_scaled = sc.fit_transform(cars_n)
cars_scaled = pd.DataFrame(cars_scaled, columns=cars_n.columns, index=cars_n.index)

#ecludean distance
help(pdist)
cars_e = pdist(cars_scaled)

#make this squareform
from scipy.spatial.distance import squareform
cars_es = squareform(cars_e)



######### distance matrix for categorical matrix

#first we need to hot encoded the data
cars_cc = cars_c.astype('category')
cars_cd = pd.get_dummies(cars_cc, drop_first=True)

##jaccard distance
cars_j = pdist(cars_cd, metric='jaccard')
cars_js = squareform(cars_j)

##check

type(cars_js)
cars_js.shape


dmat = cars_e + cars_j 
dmat_s = squareform(dmat)
