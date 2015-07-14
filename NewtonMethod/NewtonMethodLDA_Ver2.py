# -*- coding: utf-8 -*-
'''
Goal : 
Author : Yonghan Jung, ISyE, KAIST 
Date : 15
Comment 
- 

'''

''' Library '''
import numpy as np
import scipy as sp
import pandas as pd
import matplotlib.pyplot as plt
from scipy import optimize

''' Function or Class '''


class NewtonLDA:
    def __init__(self, DictArrayMatrix_Input):
        self.DictArrayMatrix_Input = DictArrayMatrix_Input
        self.DictArrayMatrix_Concatenate = np.concatenate([DictArrayMatrix_Input[0], DictArrayMatrix_Input[1]])
        self.Int_NumClass1 = len(DictArrayMatrix_Input[0])
        self.Int_NumClass2 = len(DictArrayMatrix_Input[1])
        self.Int_TotalNum = self.Int_NumClass1 + self.Int_NumClass2
        self.Int_Dim = len(self.DictArrayMatrix_Concatenate[0])
        self.Array_Class1Mean = np.mean(DictArrayMatrix_Input[0],axis=0)
        self.Array_Class2Mean = np.mean(DictArrayMatrix_Input[1],axis=0)
        self.TotalMean = np.mean(self.DictArrayMatrix_Concatenate, axis = 0)

        self.ArrayMatrix_Regularizer = np.matrix(1e-6 * np.eye(self.Int_Dim))

    ####### Fisher LDA ###########
    def BetweenScatter(self):
        ArrayColumn_Class1Mu = np.reshape(self.Array_Class1Mean - self.TotalMean, (self.Int_Dim,1))
        ArrayRow_Class1Mu = np.reshape(self.Array_Class1Mean - self.TotalMean, (1,self.Int_Dim))
        ArrayMatrix_Class1 = (self.Int_NumClass1 / float(self.Int_TotalNum)) *  ArrayColumn_Class1Mu * ArrayRow_Class1Mu

        ArrayColumn_Class2Mu = np.reshape(self.Array_Class2Mean - self.TotalMean, (self.Int_Dim,1))
        ArrayRow_Class2Mu = np.reshape(self.Array_Class2Mean - self.TotalMean, (1,self.Int_Dim))
        ArrayMatrix_Class2 = (self.Int_NumClass2 / float(self.Int_TotalNum)) * ArrayColumn_Class2Mu * ArrayRow_Class2Mu

        return np.matrix(ArrayMatrix_Class1 + ArrayMatrix_Class2)
    ### Within Matrix with small Regularization ###

    def WithinScatter(self):
        ArrayMatrix_Class1Scatter = np.zeros((self.Int_Dim,self.Int_Dim))
        ArrayMatrix_Class2Scatter = np.zeros((self.Int_Dim,self.Int_Dim))

        for Array_Class1Data in self.DictArrayMatrix_Input[0]:
            ArrayColumn_DiffClass1 = np.reshape(Array_Class1Data - self.Array_Class1Mean,(self.Int_Dim,1))
            ArrayRow_DiffClass1 = np.reshape(Array_Class1Data - self.Array_Class1Mean,(1,self.Int_Dim))
            ArrayMatrix_Class1Scatter += ArrayColumn_DiffClass1 * ArrayRow_DiffClass1
        ArrayMatrix_Class1Scatter *= (self.Int_NumClass1 / float(self.Int_TotalNum))

        for Array_Class2Data in self.DictArrayMatrix_Input[1]:
            ArrayColumn_DiffClass2 = np.reshape(Array_Class2Data - self.Array_Class2Mean,(self.Int_Dim,1))
            ArrayRow_DiffClass2 = np.reshape(Array_Class2Data - self.Array_Class2Mean,(1,self.Int_Dim))
            ArrayMatrix_Class2Scatter += ArrayColumn_DiffClass2 * ArrayRow_DiffClass2
        ArrayMatrix_Class2Scatter *= (self.Int_NumClass2 / float(self.Int_TotalNum))
        return np.matrix(ArrayMatrix_Class1Scatter + ArrayMatrix_Class2Scatter)

    def SQRTInverseMatrix(self, MyArray):
        EigVal, EigMat = np.linalg.eigh(MyArray)
        EigDiag = np.eye(len(EigVal))
        # print EigDiag

        EigMat = np.matrix(EigMat)
        for idx in range(len(EigVal)):
            EigDiag[idx][idx] = (np.sqrt(EigVal[idx]))
        EigDiag = np.matrix(EigDiag)
        return EigMat * EigDiag * (EigMat.I)

    def FisherLDA(self):
        S_W = self.WithinScatter()
        S_B = self.BetweenScatter()
        SQRT_S_W = self.SQRTInverseMatrix(S_W)
        EigVal, EigMat = np.linalg.eigh(SQRT_S_W * S_B * SQRT_S_W)
        W = SQRT_S_W * EigMat
        IDX = np.argsort(EigVal)[::-1]
        EigMat = EigMat[:,IDX]
        W = EigMat[:,:self.Int_Dim]
        ## Columnwise Normalize
        # W = np.squeeze(np.asarray(W))
        # RowSum = np.zeros(self.Int_Dim)
        # for RowIdx in range(self.Int_Dim):
        #     RowSum += np.abs(W[RowIdx])
        # for RowIdx in range(self.Int_Dim):
        #     W[RowIdx] /= RowSum
        return W


    ####### Gradient Descent ###########
    # 0. Matrix Norm
    # 1. Objective Function
    # 2. Gradient Function
    # 3. Gradient Descent

    # Function to be minimized

    def MatrixNorm(self,MATRIX):
        MATRIX = np.matrix(MATRIX)
        return np.sqrt(np.trace(MATRIX * MATRIX.T))

    def ObjectiveFunction(self,W):
        W = np.reshape(W, (self.Int_Dim, self.Int_Dim))
        Lambda = 0
        Sb = self.BetweenScatter()
        Sb = np.reshape(Sb, (self.Int_Dim, self.Int_Dim))
        Sw = self.WithinScatter()
        Sw = np.reshape(Sw, (self.Int_Dim, self.Int_Dim))
        WT = np.transpose(W)
        try :
            return -1 * (np.trace((WT * Sb * W) * (np.linalg.inv(WT * Sw * W)))) + Lambda*np.linalg.norm(W,np.infty)
        except:
            return -1 * (np.trace((WT * Sb * W) * (np.linalg.inv(WT * Sw * W + self.ArrayMatrix_Regularizer)))) + Lambda*np.linalg.norm(W,np.infty)

    def Constraint(self,W):
        W = np.reshape(W, (self.Int_Dim, self.Int_Dim))
        MaxVal = 0
        for row in W :
            MaxVal += max(abs(row))
        return MaxVal - 10

    def GradientFunction(self,W):
        W = np.reshape(W, (self.Int_Dim, self.Int_Dim))
        Sb = self.BetweenScatter()
        Sb = np.reshape(Sb,(self.Int_Dim, self.Int_Dim))
        Sw = self.WithinScatter()
        Sw = np.reshape(Sw, (self.Int_Dim, self.Int_Dim))
        WT = np.transpose(W)
        MatReg = np.reshape(self.ArrayMatrix_Regularizer, (self.Int_Dim, self.Int_Dim))

        try :
            FirstTerm = -1 * (Sb * W * (np.linalg.inv(WT * Sw * W)))
            SecondTerm = (Sw * W) * (np.linalg.inv(WT * Sw * W))
            ThirdTerm = (WT * Sb * W) * (np.linalg.inv(WT * Sw * W))
            return FirstTerm + (SecondTerm * ThirdTerm)
        except:
            FirstTerm = -1 * (Sb * W * (np.linalg.inv(WT * Sw * W + MatReg)))
            SecondTerm = (Sw * W) * (np.linalg.inv(WT * Sw * W+ MatReg))
            ThirdTerm = (WT * Sb * W) * (np.linalg.inv(WT * Sw * W+ MatReg))
            return FirstTerm + (SecondTerm * ThirdTerm)

    def LDATransformation(self, W):
        W = np.matrix(W)
        DictArrayMatrix_NewTransformed = dict()
        DictArrayMatrix_NewTransformed[0] = np.squeeze(np.asarray(np.matrix(self.DictArrayMatrix_Input[0]) * W ))
        DictArrayMatrix_NewTransformed[1] = np.squeeze(np.asarray(np.matrix(self.DictArrayMatrix_Input[1]) * W ))
        return DictArrayMatrix_NewTransformed

    def PLOT(self, DictArrayMatrix, TITLE):
        ArrayMatrix_0 = DictArrayMatrix[0]
        ArrayMatrix_1 = DictArrayMatrix[1]

        plt.figure()
        plt.title(TITLE)
        plt.grid()
        for each in ArrayMatrix_0:
            for idx, val in enumerate(each):
                plt.plot(idx, val, 'bo')

        for each in ArrayMatrix_1:
            for idx, val in enumerate(each):
                plt.plot(idx, val, 'ro')



# Two class
# Row : Data record
# Col : Data Column
def GeneratingSimulationData(Int_Seed, Int_Mu1, Int_Mu2, Int_Dim, Int_DataNum):
    np.random.seed(Int_Seed)
    DictArrayMatrix = dict()
    Array_Mu1 = np.array([Int_Mu1] * Int_Dim)
    ArrayMatrix_Cov1 = np.eye(Int_Dim)
    DictArrayMatrix[0] = np.random.multivariate_normal(Array_Mu1, ArrayMatrix_Cov1, Int_DataNum)

    Array_Mu2 = np.array([Int_Mu2] * Int_Dim)
    ArrayMatrix_Cov2 = 2 * np.eye(Int_Dim)
    DictArrayMatrix[1] = np.random.multivariate_normal(Array_Mu2, ArrayMatrix_Cov2, Int_DataNum/2)
    return DictArrayMatrix



if __name__ == "__main__":
    Int_Seed = 1
    Int_Mu1 = 1
    Int_Mu2 = -1
    Int_Dim = 5
    Int_DataNum = 100

    DictArrayMatrix_Data = GeneratingSimulationData(Int_Seed = Int_Seed, Int_Mu1 = Int_Mu1, Int_Mu2 = Int_Mu2, Int_Dim = Int_Dim, Int_DataNum=Int_DataNum)
    DictArrayMatrix_TotalData = np.concatenate([DictArrayMatrix_Data[0], DictArrayMatrix_Data[1]])

    # Class
    Object_NewtonLDA = NewtonLDA(DictArrayMatrix_Input=DictArrayMatrix_Data)
    W = Object_NewtonLDA.FisherLDA()

    result = optimize.fmin_cobyla(func = Object_NewtonLDA.ObjectiveFunction, x0=W, cons=Object_NewtonLDA.Constraint)
    # result = optimize.fmin(func=Object_NewtonLDA.ObjectiveFunction, x0=W, disp=1)
    New_W = result.reshape(Int_Dim,Int_Dim)

    LDATransformation = Object_NewtonLDA.LDATransformation(W)
    CVXTransformation = Object_NewtonLDA.LDATransformation(New_W)

    print Object_NewtonLDA.ObjectiveFunction(W)
    print Object_NewtonLDA.ObjectiveFunction(New_W)
    print pd.DataFrame(W)
    print pd.DataFrame(New_W)
    print ""

    for dim in range(Int_Dim):
        print dim, max(abs(New_W[:,dim]))

    Object_NewtonLDA.PLOT(LDATransformation, "Fisher LDA")
    Object_NewtonLDA.PLOT(CVXTransformation, "CVX LDA")
    plt.show()

    # print Object_NewtonLDA.ObjectiveFunction(W)
    # print Object_NewtonLDA.ObjectiveFunction(New_W)
    # print W
    # print New_W



