# -*- coding: utf-8 -*-
'''
Goal : Hans KMenas
Author : Yonghan Jung, ISyE, KAIST 
Date : 150507
Comment 
- One row one data record
-

'''

''' Library '''
import numpy as np
import matplotlib.pyplot as plt
''' Function or Class '''

# Data : n by p feature data
class KMeans :
    def __init__(self, Data, K):
        self.data = np.array(Data, dtype='float32')
        self.K = K
        self.RowNum = self.data.shape[0]
        self.ColNum = self.data.shape[1]
        if self.data.shape[0] < self.data.shape[1] :
            self.data = self.data.transpose()


    def InitialCluster(self):
        Centroid = self.data[np.random.choice(self.RowNum, self.K, replace = False)]
        ResultDict = dict()
        for idx in range(self.K):
            ResultDict[idx] = Centroid[idx]

        for datarow in self.data:
            ClusterIdx = np.argmin(np.sum((datarow - Centroid) ** 2, axis=1))
            if datarow not in ResultDict[ClusterIdx]:
                ResultDict[ClusterIdx] = np.vstack([ResultDict[ClusterIdx], datarow])
            else:
                pass
        return Centroid, ResultDict

    def Cluster(self):
        Cent, ResultDict = self.InitialCluster()
        for idx in range(100):
            PrevCent = Cent
            for idx, key in enumerate(ResultDict):
                Cent[key] = np.mean(ResultDict[key], axis=0)
            ResultDict = dict()
            for idx in range(self.K):
                ResultDict[idx] = Cent[idx]

            for datarow in self.data:
                ClusterIdx = np.argmin(np.sum((datarow - Cent) ** 2, axis=1))
                if datarow not in ResultDict[ClusterIdx]:
                    ResultDict[ClusterIdx] = np.vstack([ResultDict[ClusterIdx], datarow])
                else:
                    pass
            if np.sqrt(np.sum((Cent - PrevCent) ** 2)) < 0.0001:
                break

        return Cent, ResultDict




    # def Data_A

def Random_Data_Generator(dim, mu1, mu2, mu3):
    np.random.seed(19230)
    Mu1 = np.array([mu1] * dim)
    COV1 = np.eye(dim)
    # It is common to arrange data in column form
    DataC1 = np.random.multivariate_normal(Mu1, COV1, 200).T

    Mu2 = np.array([mu2] * dim)
    COV2 = np.eye(dim)
    DataC2 = np.random.multivariate_normal(Mu2, COV2, 200).T

    Mu3 = np.array([mu3] * dim)
    COV3 = np.eye(dim)
    DataC3 = np.random.multivariate_normal(Mu3, COV3, 200).T


    Data = np.concatenate([DataC1,DataC2, DataC3], axis=1)

    return Data.T


if __name__ == "__main__":
    Data = Random_Data_Generator(2,-0.1,0,0.1)
    print Data
    # Data = np.random.rand(1000,2)
    KM = KMeans(Data,3)
    Cent, Cluster =  KM.Cluster()
    InitCent,Initcluster = KM.InitialCluster()

    # print Cent
    # print ""
    # print Cluster

    plt.figure()
    plt.plot(Data[:,0], Data[:,1],'bo')

    plt.figure()
    color = ['ro','go','bo','wo']
    for idx, key in enumerate(Cluster):
        plt.title("CLUSTER")
        plt.plot(Cluster[key][:,0], Cluster[key][:,1], color[key])
    plt.figure()
    for idx, key in enumerate(Initcluster):
        plt.title("INIT")
        plt.plot(Initcluster[key][:,0], Initcluster[key][:,1], color[key])
    plt.show()