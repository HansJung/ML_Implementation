# -*- coding: utf-8 -*-
'''
Goal : Simulate via PCA
Author : Yonghan Jung, ISyE, KAIST 
Date : 150502
Comment 
- 

'''

''' Library '''
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from sklearn.decomposition import PCA

from PCA import Hans_PCA


''' Function or Class '''
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
    dim = 4
    Data = Random_Data_Generator(dim, 0, 10)

    MyPCA = Hans_PCA(Data)
    pca = PCA(n_components=dim)
    ComPCA = pca.fit_transform(Data)

    print Data
    print MyPCA.PCA_Transform()
    print MyPCA.Dimension_Reduction(2)







