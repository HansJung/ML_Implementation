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
from sklearn.utils.extmath import logsumexp
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

    # def SampleVar(self, X, Mu):
    #     X = np.array(X)
    #     rownum = X.shape[0]
    #     Mu = np.array(Mu)
    #     return np.dot((X - Mu).T, (X - Mu))/rownum

    # def logsumexp(arr, axis=0):
    #     arr = np.rollaxis(arr, axis)
    #     # Use the max to normalize, as with the log this is what accumulates
    #     # the less errors
    #     vmax = arr.max(axis=0)
    #     out = np.log(np.sum(np.exp(arr - vmax), axis=0))
    #     out += vmax
    #     return out

    def Initial_Setting(self):
        KMeansObj = KMeans(self.data, self.K)
        InitialMean, ClusterDict = KMeansObj.Cluster()
        SampleVar = np.cov(self.data.T) + self.min_covar * np.eye(self.data.shape[1])
        SampleVar = np.tile(np.diag(SampleVar), (self.K,1))
        InitialAssign = np.tile(1.0 / self.K,
                                        self.K)

        return InitialMean, SampleVar, InitialAssign

    def LogMultiNormalDensity(self, means, covars, weights):
        n_samples, n_dim = self.data.shape
        lpr = -0.5 * (n_dim * np.log(2 * np.pi) + np.sum(np.log(covars), 1)
                      + np.sum((means ** 2) / covars, 1)
                      - 2 * np.dot(self.data, (means / covars).T)
                      + np.dot(self.data ** 2, (1.0 / covars).T))
        return lpr

    def ComputeGMM_LR(self, means, covars, weights):
        lpr = (self.LogMultiNormalDensity(means, covars, weights)
               + np.log(weights))
        logprob = logsumexp(lpr, axis=1)
        responsibilities = np.exp(lpr - logprob[:, np.newaxis])
        return logprob, responsibilities

    def MaxStep(self, Responsibility):
        EPS = np.finfo(float).eps
        weights = Responsibility.sum(axis=0)
        weighted_X_sum = np.dot(Responsibility.T, self.data)
        inverse_weights = 1.0 / (weights[:, np.newaxis] + 10 * EPS)
        MyWeights = (weights / (weights.sum() + 10 * EPS) + EPS)
        MyMeans = weighted_X_sum * inverse_weights

        MyCovs = self._covar_mstep_diag(self.data, Responsibility, weighted_X_sum, inverse_weights, self.min_covar )

        return MyMeans, MyCovs, MyWeights



    def _covar_mstep_diag(self, X, responsibilities, weighted_X_sum, norm,
                      min_covar):
        EPS = np.finfo(float).eps
        weights = responsibilities.sum(axis=0)
        Means = np.dot(responsibilities.T, self.data) * (1.0 / (weights[:, np.newaxis] + 10 * EPS))

        avg_X2 = np.dot(responsibilities.T, X * X) * norm
        avg_means2 = Means ** 2
        avg_X_means = Means * weighted_X_sum * norm
        return avg_X2 - 2 * avg_X_means + avg_means2 + min_covar


    def EMIter(self):
        means, covars, weights = self.Initial_Setting()
        LogLike = []
        NumIter = 100
        max_log_prob = -np.infty
        for i in range(NumIter):
            curr_log_likelihood, responsibilities = self.ComputeGMM_LR(means, covars, weights)
            LogLike.append(curr_log_likelihood.sum())
            if i > 0 and abs(LogLike[-1] - LogLike[-2]) < 1e-6:
                break
            means, covars, weights = self.MaxStep(responsibilities)

            if LogLike[-1] > max_log_prob:
                max_log_prob = LogLike[-1]
                break

        return means, covars, weights

    def GMM_Cluster(self):
        ResultDict = dict()
        mean, covar, weights = self.EMIter()
        logprob, responsibilities = self.ComputeGMM_LR(mean, covar, weights)
        keylist = responsibilities.argmax(axis=1)
        for idx, datarow in enumerate(self.data):
            key = keylist[idx]
            if key in ResultDict.keys():
                ResultDict[key] = np.vstack([ResultDict[key], datarow])
            else:
                ResultDict[key] = datarow
        return ResultDict







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
    K = 3
    MyEM = EM_GMM(Data,K)
    # ResultData = MyEM.ComputeEM()
    Mu,Var,Weight = MyEM.EMIter()
    ResultData = MyEM.GMM_Cluster()


    print Weight, Mu


    g = mixture.GMM(n_components=3)
    g.fit(Data)
    print "Ans", g.weights_, g.means_


    # print "ANSWER", g.means_, g.weights_

    #
    #
    plt.figure()
    plt.plot(Data[:,0], Data[:,1],'bo')
    color = ['ro','go','bo']
    plt.figure()
    for idx, key in enumerate(ResultData):
        plt.title("CLUSTER")
        plt.plot(ResultData[key][:,0], ResultData[key][:,1], color[key])



    plt.show()