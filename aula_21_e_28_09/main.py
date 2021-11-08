import pandas as pd
import seaborn as sns

from matplotlib import rcParams

from sklearn import datasets
from sklearn.decomposition import PCA

rcParams['figure.figsize'] =4,5
sns.set_style('whitegrid')

iris= datasets.load_iris()

x=iris.data
nomes_das_variaveis = iris.feature_names
x[0:5,]

pca = PCA()
iris_pca = pca.fit_transform(x)

pca.explained_variance_ratio_

pca.explained_variance_ratio_.sum()

comps = pd.DataFrame(pca.components_, columns=nomes_das_variaveis)
comps

sns.heatmap(comps)