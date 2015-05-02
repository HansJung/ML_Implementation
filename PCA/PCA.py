# -*- coding: utf-8 -*-
'''
Goal : To implement PCA
Author : Yonghan Jung, ISyE, KAIST
Date : 150502
Comment 
- Input
: Data in numpy array (Each "row" represents a record)
: Equivalently, Each column represents a type of each features

'''

''' Library '''
import numpy as np
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt

''' Function or Class '''
class Hans_PCA:
    def __init__(self, Data):
        self.Data = np.asarray(Data, dtype='float32')
        self.ColMean = np.mean(Data.T, axis = 1, dtype='float32')
        self.CenterData = self.Data - self.ColMean # All column means are zeros
        self.Scatter = self.CenterData.T.dot(self.CenterData)

    def EigMatScatter(self):
        EigVals, EigMat = np.linalg.eig(self.Scatter)
        idx = EigVals.argsort()[::-1]
        EigVals = EigVals[idx]
        EigMat = EigMat[:,idx]
        return EigVals, EigMat

    def PCA_Transform(self):
        EigVals, EigMat = self.EigMatScatter()
        print EigVals
        return np.dot(self.CenterData, EigMat )

    def PCA_Explained_Ratio(self):
        EigVals, EigMat = self.EigMatScatter()
        return np.array([x / float(sum(EigVals)) for x in EigVals ], dtype='float32')


def Random_Data_Generator(dim, mu1, mu2):
    np.random.seed(234234782384239784)
    Mu1 = np.array([mu1] * dim)
    COV1 = np.eye(dim)
    # It is common to arrange data in column form
    DataC1 = np.random.multivariate_normal(Mu1, COV1, 20).T

    Mu2 = np.array([mu2] * dim)
    COV2 = np.eye(dim)
    DataC2 = np.random.multivariate_normal(Mu2, COV2, 20).T
    Data = np.concatenate([DataC1,DataC2], axis=1)

    return Data.T


if __name__ == "__main__":
    ''' Random two classes data generated '''
    dim = 5
    Data = Random_Data_Generator(dim, 0, 10)

    MyPCA = Hans_PCA(Data)
    pca = PCA(n_components=dim)
    ComPCA = pca.fit_transform(Data)

    print MyPCA.PCA_Explained_Ratio()
    print pca.explained_variance_ratio_

    # print pca.fit_transform(Data)
    plt.figure()
    plt.plot(Data[:,0],Data[:,1], 'bo')
    plt.grid()

    PCAData = MyPCA.PCA_Transform()
    plt.figure()
    plt.grid()
    plt.plot(PCAData[:,0], PCAData[:,1], 'bo')

    plt.figure()
    plt.grid()
    plt.plot(ComPCA[:,0], ComPCA[:,1], 'bo')
    plt.show()