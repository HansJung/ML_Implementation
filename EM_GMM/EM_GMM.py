# -*- coding: utf-8 -*-
'''
Goal : EM Algorithm
Author : Yonghan Jung, ISyE, KAIST
Date : 150507
Comment
-

'''

''' Library '''
import numpy as np
import matplotlib.pyplot as plt
from KMeans.HansKMeans import KMeans
from PDFs.PDF import PDF
from sklearn import mixture
''' Function or Class '''


class EM_GMM:
    def __init__(self, DataTrain, K):
        self.data = DataTrain
        self.K = K
        self.RowNum = self.data.shape[0]
        self.ColNum = self.data.shape[1]
        self.min_covar = 1e-3
        if self.data.shape[0] < self.data.shape[1] :
            self.data = self.data.transpose()


def Random_Data_Generator(dim, mu1, mu2, mu3, Num):
    np.random.seed(238)
    Mu1 = np.array([mu1] * dim)
    COV1 = np.eye(dim)
    # It is common to arrange data in column form
    DataC1 = np.random.multivariate_normal(Mu1, COV1, Num).T

    Mu2 = np.array([mu2] * dim)
    COV2 = np.eye(dim)
    DataC2 = np.random.multivariate_normal(Mu2, COV2, Num).T

    Mu3 = np.array([mu3] * dim)
    COV3 = np.eye(dim)
    DataC3 = np.random.multivariate_normal(Mu3, COV3, Num).T

    Data = np.concatenate([DataC1,DataC2, DataC3], axis=1)

    return Data.T

if __name__ == "__main__":
    Data = Random_Data_Generator(2,-10,0,10, 100)

    # Plot
    plt.scatter(Data[:,0], Data[:,1])
    plt.show()