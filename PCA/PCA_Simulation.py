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


''' Function or Class '''

if __name__ == "__main__":
    ''' Random two classes data generated '''
    np.random.seed(234234782384239784)
    # Two classes
    Mu1 = np.array([0] * 3)
    COV1 = np.eye(3)
    # It is common to arrange data in column form
    DataC1 = np.random.multivariate_normal(Mu1, COV1, 20).T

    Mu2 = np.array([0] * 3)
    COV2 = np.eye(3)
    DataC2 = np.random.multivariate_normal(Mu2, COV2, 20).T
    Data = np.concatenate([DataC1,DataC2], axis=1)

    print pd.DataFrame(Data)





