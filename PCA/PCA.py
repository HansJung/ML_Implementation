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
        return np.dot(self.CenterData, EigMat)

    def PCA_Explained_Ratio(self):
        EigVals, EigMat = self.EigMatScatter()
        return np.array([x / float(sum(EigVals)) for x in EigVals ], dtype='float32')

    def Dimension_Reduction(self, dim):
        PCAData = self.PCA_Transform()
        return PCAData[:,:dim]


