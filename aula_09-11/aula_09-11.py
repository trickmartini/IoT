import numpy as np
import pandas as pd
import sys
import matplotlib.pyplot as plt
from scipy.signal import savgol_filter
from sklearn.decomposition import PCA
import warnings

warnings.filterwarnings('ignore')

def plot(x, y):
    with plt.style.context(('ggplot')):
        plt.plot(x, y)
        plt.xlabel(u'Wavelength')
        plt.ylabel(u'Intensisty')
        plt.title(u'Spectra chart')
        plt.show()


def snv(input_data):
    # Define a new array and populate it with the corrected data
    output_data = np.zeros_like(input_data)
    for i in range(input_data.shape[0]):
        # Apply correction
        output_data[i, :] = (input_data[i, :] - np.mean(input_data[i, :])) / np.std(input_data[i, :])
    return output_data


df = pd.read_csv("nir.csv")
print(df)

wave = df.iloc[1:, 0].to_numpy()
print(wave)

# sys.exit()
data = df.iloc[1:, 1:].to_numpy().T
print(data)

classes = df.iloc[0, 1:].to_numpy()
print(classes)

# plot(wave,data.T)

# sys.exit()

# não-supervisionado
print('PCA')
pca = PCA(n_components=3)
scores = pca.fit_transform(data)
# scatter(scores, classes)

data = data[:, 2857:2909]
pca = PCA(n_components=4)
data = savgol_filter(snv(data), 5, polyorder=2, deriv=1)
scores = pca.fit_transform(data)
