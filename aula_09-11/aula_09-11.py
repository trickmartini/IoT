import numpy as np
import pandas as pd
from math import floor
import sys
import matplotlib.pyplot as plt
from scipy.signal import savgol_filter
from sklearn.decomposition import PCA
import warnings
import getopt

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
        print(len(output_data))
    return output_data


def scatter(scores, classes, title):
    unique = list(set(classes))
    colors = [plt.cm.jet(float(i) / max(unique)) for i in unique]
    with plt.style.context(('ggplot')):
        for i, u in enumerate(unique):
            xi = [scores[j, 0] for j in range(len(scores[:, 0])) if classes[j] == u]
            yi = [scores[j, 1] for j in range(len(scores[:, 1])) if classes[j] == u]
            plt.scatter(xi, yi, c=colors[i], s=60, edgecolors='k', label=str(u))
        plt.xlabel('PC1')
        plt.ylabel('PC2')
        # plt.legend(labplot,loc='lower right')
        plt.title(f'Principal Component Analysis - {title}')
    plt.show()


def calcula_intervalo(filePath, intervalos):
    df = pd.read_csv(filePath)
    classes = df.iloc[0, 1:].to_numpy()
    data = df.iloc[1:, 1:].to_numpy().T
    print(len(data[0]))
    print(data[:, 209:418])
    lenDF = floor((len(df) - 1) / intervalos)

    for i in range(intervalos):
        if i == 0:
            init = i * lenDF
        else:
            init = i * lenDF + 1
        if i == intervalos - 1:
            final = len(data[0])
        else:
            final = init + lenDF
        dataAux = data[:, init:final]

        pca = PCA(n_components=4)
        # data = savgol_filter(snv(data), 5, polyorder=2, deriv=1)
        scores = pca.fit_transform(dataAux)
        titleAux = str(init) + ' - ' + str(final)
        scatter(scores, classes, titleAux)


# pega argumentos linha comando

filePath = ""
interval = ""
argv = sys.argv[1:]

try:
    options, args = getopt.getopt(argv, "f:i:",
                                  ["file =",
                                   "interval ="])
except:
    print("Error Message ")

for name, value in options:
    if name in ['-f', '--file']:
        filePath = value
    elif name in ['-i', '--interval']:
        interval = value

calcula_intervalo(filePath, int(interval))

# "aula_09-11/nir.csv"

# py .\aula_09-11\aula_09-11.py -f "aula_09-11/nir.csv" -i 3
