# -*- coding: utf-8 -*-
'''
Goal : To impement the PDF
Author : Yonghan Jung, ISyE, KAIST 
Date : 150504
Comment 
- 

'''

''' Library '''
import numpy as np
import math
import operator
from scipy.stats import binom
from scipy.stats import beta
from scipy.stats import norm
from scipy.optimize import minimize

''' Function or Class '''


class PDF:
    def __init__(self):
        pass

    def Norm(self,X,Mu,Var):
        return norm.pdf(X,Mu,Var)

    def MultiNorm(self,X, Mu, Cov):
        X = np.asarray(X, dtype='float32')
        Dimension = len(X)
        Mu = np.asarray(Mu, dtype='float32')
        Cov = np.matrix(Cov)
        if Dimension == len(Mu) and (Dimension, Dimension) == Cov.shape:
            det = np.linalg.det(Cov)
            if det == 0:
                raise NameError("The covariance matrix can't be singular")
            norm_const = 1.0/ ( math.pow((2*math.pi),float(Dimension)/2) * math.pow(det,1.0/2) )
            X_Mu = np.matrix(X - Mu)
            InvCov = Cov.I
            result = math.pow(math.e, -0.5 * (X_Mu * InvCov * X_Mu.T))
            return norm_const * result
        else:
            raise NameError("The dimensions of the input don't match")

    def Binomial(self, X, N, P):
        Params = [N,P]
        return binom.pmf(X,N,P)

    def Beta(self, X, a,b):
        return beta.pdf(X,a,b )

    def Dirichlet(self,x, alpha):
        x = np.asarray(x, dtype='float32')
        alpha = np.asarray(alpha, dtype='float32')
        return (math.gamma(sum(alpha)) /
          reduce(operator.mul, [math.gamma(a) for a in alpha]) *
          reduce(operator.mul, [x[i]**(alpha[i]-1.0) for i in range(len(alpha))]))

    def Normal_MLE(self, Obs, Mu, Cov):
        def Normal_LogLikelihood(Params):
            Mu = Params[0]
            Var = Params[1]
            LogLikelihood = -np.sum(norm.logpdf(Obs,loc=Mu,scale=Var))
            return LogLikelihood
        Init_Param = [Mu, Cov]
        return minimize(Normal_LogLikelihood, Init_Param, method='nelder-mead').x



if __name__ == "__main__":
    Cov = np.array([[2.3, 0, 0, 0],
           [0, 1.5, 0, 0],
           [0, 0, 1.7, 0],
           [0, 0,   0, 2]
          ], dtype='float32')
    mu = np.array([2,3,8,10], dtype='float32')
    x = np.array([2.1,3.5,8, 9.5])

    MyPDF = PDF()
    alpha = [1,20,3]

    init = [1,1]
    print MyPDF.Normal_MLE(x, 0, 1)








