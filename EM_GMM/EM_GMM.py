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
        if self.data.shape[0] < self.data.shape[1] :
            self.data = self.data.transpose()

    # def SampleVar(self, X, Mu):
    #     X = np.array(X)
    #     rownum = X.shape[0]
    #     Mu = np.array(Mu)
    #     return np.dot((X - Mu).T, (X - Mu))/rownum

    def Initial_Setting(self):
        KMeansObj = KMeans(self.data, self.K)
        InitialMean, ClusterDict = KMeansObj.Cluster()
        SampleVar = dict()
        for idx, key in enumerate(ClusterDict):
            SampleVar[key] = np.cov(ClusterDict[key].T)
        Initial_Assign = np.array([1/float(K)] * K)
        return InitialMean, SampleVar, Initial_Assign

    def InitialGMM(self):
        MyPDF = PDF()
        Mu, Cov, Asgn = self.Initial_Setting()
        ResultDict = dict()
        for datarow in self.data:
            # ClusterIdx = np.argmin(np.sum((datarow - Cent) ** 2, axis=1))
            PDFClass = np.array([MyPDF.MultiNorm(datarow, Mu[key], Cov[key]) for key in Cov.keys()])
            if np.argmax(PDFClass) in ResultDict.keys() :
                ResultDict[np.argmax(PDFClass)] = np.vstack([ResultDict[np.argmax(PDFClass)], datarow])
            else:
                ResultDict[np.argmax(PDFClass)] = np.array(datarow)
        return ResultDict

    def ComputeParam(self, ResultDict):
        Means = []
        Covs = dict()
        Asgn = []

        for idx, key in enumerate(ResultDict):
            Mean = np.mean(ResultDict[key], axis=0)
            Covs[key] = np.cov(ResultDict[key].T)
            Means.append(Mean)
            Asgn.append(len(ResultDict[key]) / float(self.RowNum))
        Means = np.array(Means)
        Asgn = np.array(Asgn)

        return Means,Covs, Asgn

    def ComputeEM(self):
        MyPDF = PDF()
        ResultDict = self.InitialGMM()
        for idx in range(20):
            Means, Covs, Asgn = self.ComputeParam(ResultDict)
            ResultDict = dict()
            for datarow in self.data:
                # ClusterIdx = np.argmin(np.sum((datarow - Cent) ** 2, axis=1))
                PDFClass = np.array([MyPDF.MultiNorm(datarow, Means[key], Covs[key]) * Asgn[key] for key in Covs.keys()])
                if np.argmax(PDFClass) in ResultDict.keys() :
                    ResultDict[np.argmax(PDFClass)] = np.vstack([ResultDict[np.argmax(PDFClass)], datarow])
                else:
                    ResultDict[np.argmax(PDFClass)] = np.array(datarow)
            print idx, Means, Asgn
        return ResultDict







def Random_Data_Generator(dim, mu1, mu2, mu3):
    np.random.seed(567)
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
    Data = Random_Data_Generator(2,-1,0,1)
    K = 3
    MyEM = EM_GMM(Data,K)
    Mu, Var, InitialAssign = MyEM.Initial_Setting()
    ResultData = MyEM.ComputeEM()

    g = mixture.GMM(n_components=3)
    g.fit(Data)
    print "ANSWER", g.means_, g.weights_



    plt.figure()
    plt.plot(Data[:,0], Data[:,1],'bo')
    color = ['ro','go','bo']
    plt.figure()
    for idx, key in enumerate(ResultData):
        plt.title("CLUSTER")
        plt.plot(ResultData[key][:,0], ResultData[key][:,1], color[key])



    plt.show()