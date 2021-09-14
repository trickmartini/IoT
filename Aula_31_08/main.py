import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

if __name__ == '__main__':
    # read matrix file
    matrix = pd.read_csv('./aula4-matriz.csv', header=None, sep=',', lineterminator='\n')

    # calculate meancenter and autoscale
    meanCenterMatrix = matrix - matrix.mean()
    AutoScaleMatrix = (matrix - matrix.mean())/matrix.std()

    # convert Dataframes to numpy arrays, to use on charts
    numArrayMatrix = matrix.to_numpy()
    numArraymeanCenterMatrix = meanCenterMatrix.to_numpy()
    numArrayAutoScaleMatrix = AutoScaleMatrix.to_numpy()

    # plot charts
    figure = plt.figure()
    plt.subplots()
    plt.plot(numArrayMatrix, numArraymeanCenterMatrix, numArrayAutoScaleMatrix)
    plt.show()

