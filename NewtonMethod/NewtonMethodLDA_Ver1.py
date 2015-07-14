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
import matplotlib.pyplot as plt
from scipy import optimize
import pandas as pd
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

        self.ArrayMatrix_Regularizer = 1e-6 * np.eye(self.Int_Dim)

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

    def OriginalLDA(self):
        S_W = self.WithinScatter()
        S_B = self.BetweenScatter()
        SQRT_S_W = self.SQRTInverseMatrix(S_W)
        _, EigMat = np.linalg.eigh(SQRT_S_W * S_B * SQRT_S_W)
        W = SQRT_S_W * EigMat
        ## Columnwise Normalize
        W = np.squeeze(np.asarray(W))
        RowSum = np.zeros(self.Int_Dim)
        for RowIdx in range(self.Int_Dim):
            RowSum += np.abs(W[RowIdx])
        for RowIdx in range(self.Int_Dim):
            W[RowIdx] /= RowSum
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
        W = np.matrix(W)
        Sb = self.BetweenScatter()
        Sw = self.WithinScatter()
        try :
            return -1 * np.trace((W.T * Sb * W) * ((W.T * Sb * W).I))
        except:
            return -1 * np.trace((W.T * Sb * W) * ((W.T * Sb * W + self.ArrayMatrix_Regularizer).I))

    def GradientFunction(self,W):
        Sb = self.BetweenScatter()
        Sw = self.WithinScatter()
        W = np.matrix(W)

        try :
            FirstTerm = -1 * (Sb * W * ((W.T * Sw * W).I))
            SecondTerm = (Sw * W) * ((W.T * Sw * W).I)
            ThirdTerm = (W.T * Sb * W) * ((W.T * Sw * W).I)
            return np.matrix(FirstTerm + (SecondTerm * ThirdTerm))
        except:
            FirstTerm = -1 * (Sb * W * ((W.T * Sw * W + self.ArrayMatrix_Regularizer).I))
            SecondTerm = (Sw * W) * ((W.T * Sw * W + self.ArrayMatrix_Regularizer).I)
            ThirdTerm = (W.T * Sb * W) * ((W.T * Sw * W + self.ArrayMatrix_Regularizer).I)
            return np.matrix(FirstTerm + (SecondTerm * ThirdTerm))


    def GradientDescent(self):

        W = np.matrix(np.eye(self.Int_Dim))
        StepSize = np.array(range(1,11))/20.0

        Int_Iter = 0
        while 1:
            OptimalStep = []
            PrevW = W
            GradientMatrix = self.GradientFunction(W)
            for stepsize in StepSize:
                OptimalStep.append(self.ObjectiveFunction(W - stepsize * GradientMatrix))
            Temp_OptimalStep = StepSize[np.argmin(OptimalStep)]
            W = W - Temp_OptimalStep * GradientMatrix
            ConvergenceDistance = self.MatrixNorm(W - PrevW)
            print Int_Iter, "th Cvg", ConvergenceDistance, " | ", self.ObjectiveFunction(W)
            if ConvergenceDistance < 1e-3 or Int_Iter > 300:
                break
            Int_Iter += 1
        return W



    ######### LDA to LASSO ###########
    def LambdaVector(self, Lambda, Idx):
        Vector = np.zeros(self.Int_Dim)
        Vector[Idx] = Lambda
        return Vector

    def SoftThreshold(self, Value, Lambda):
        def Sgn(Val):
            if Val > 0 :
                return 1
            elif Val < 0:
                return -1
            elif Val == 0:
                return 0

        TargetVal = np.abs(Value) - Lambda
        if TargetVal > 0:
            return Sgn(Value) * TargetVal
        else:
            return 0


    def LASSO_Soft(self, Lambda):
        # Columnwisely
        NewW = []
        # W = self.OriginalLDA()
        W = self.GradientDescent()

        for ColIdx in range(self.Int_Dim):
            EachColumn = np.squeeze(np.asarray(W.T[ColIdx]))
            Max_WIdx = np.argmax(np.abs(EachColumn))
            Max_EachColumn = EachColumn[Max_WIdx]

            if Max_EachColumn > Lambda:
                New_Wi = EachColumn - self.LambdaVector(Lambda, Max_WIdx)
            elif Max_EachColumn < -Lambda:
                New_Wi = EachColumn + self.LambdaVector(Lambda, Max_WIdx)
            elif Max_EachColumn >= -Lambda and Max_EachColumn < Lambda:
                New_Wi = np.zeros(self.Int_Dim)
            NewW.append(New_Wi)
        NewW = np.array(NewW).T
        return NewW




    def LASSOtoLDA(self, Lambda):
        # Columnwisely
        NewW = []
        W = self.OriginalLDA()
        GradientMatrix = self.GradientFunction(W)

        for ColIdx in range(self.Int_Dim):
            EachColumn = np.squeeze(np.asarray(W.T[ColIdx]))
            EachDelta = np.squeeze(np.asarray(GradientMatrix.T[ColIdx]))

            Max_WIdx = np.argmax(np.abs(EachColumn))
            InftyNorm_W_EachColumn = np.max(np.abs(EachColumn))
            Val_W_EachColumn = EachColumn[Max_WIdx]

            Max_Delta_Idx = np.argmax(np.abs(EachDelta))
            Val_EachDelta = EachDelta[Max_Delta_Idx]

            LambdaVector = self.LambdaVector(Lambda, Max_WIdx)

            if Val_W_EachColumn < 0:
                NewW_i = EachColumn + LambdaVector
            elif Val_W_EachColumn > 0:
                NewW_i = EachColumn - LambdaVector
            elif Val_W_EachColumn == 0:
                NewW_i = np.zeros(self.Int_Dim)
            NewW.append(NewW_i)
        NewW = np.array(NewW)
        NewW = NewW.T
        return NewW


    #### REAL TRANSFORMATION #####
    def GradientWTransformation(self):
        W = self.GradientDescent()
        DictArrayMatrix_NewTransformed = dict()
        DictArrayMatrix_NewTransformed[0] = np.squeeze(np.asarray(np.matrix(self.DictArrayMatrix_Input[0]) * W))
        DictArrayMatrix_NewTransformed[1] = np.squeeze(np.asarray(np.matrix(self.DictArrayMatrix_Input[1]) * W))
        return DictArrayMatrix_NewTransformed

    def FisherLDATransformation(self):
        W = self.OriginalLDA()
        DictArrayMatrix_NewTransformed = dict()
        DictArrayMatrix_NewTransformed[0] = np.squeeze(np.asarray(np.matrix(self.DictArrayMatrix_Input[0]) * W))
        DictArrayMatrix_NewTransformed[1] = np.squeeze(np.asarray(np.matrix(self.DictArrayMatrix_Input[1]) * W))
        return DictArrayMatrix_NewTransformed

    def LASSOLDATransformation(self, Lambda):
        W = self.LASSO_Soft(Lambda)
        DictArrayMatrix_NewTransformed = dict()
        DictArrayMatrix_NewTransformed[0] = np.squeeze(np.asarray(np.matrix(self.DictArrayMatrix_Input[0]) * W))
        DictArrayMatrix_NewTransformed[1] = np.squeeze(np.asarray(np.matrix(self.DictArrayMatrix_Input[1]) * W))
        return DictArrayMatrix_NewTransformed

    def PLOT(self, DictArrayMatrix, Str_Title):
        plt.figure()
        plt.grid()
        plt.title(Str_Title)

        for Class1Data in DictArrayMatrix[0]:
            for idx, eachdata in enumerate(Class1Data):
                plt.plot(idx, eachdata, 'bo')

        for Class2Data in DictArrayMatrix[1]:
            for idx, eachdata in enumerate(Class2Data):
                plt.plot(idx, eachdata, 'ro')







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

    Flt_Lambda = 0.2

    Str_GradientTitle = "Gradient Transformation"
    Str_FLDATitle = "FLDA Transformation"
    Str_LassoTitle = "Lasso Transformation"

    # Generating Data
    DictArrayMatrix_Data = GeneratingSimulationData(Int_Seed = Int_Seed, Int_Mu1 = Int_Mu1, Int_Mu2 = Int_Mu2, Int_Dim = Int_Dim, Int_DataNum=Int_DataNum)
    DictArrayMatrix_TotalData = np.concatenate([DictArrayMatrix_Data[0], DictArrayMatrix_Data[1]])

    # Class
    Object_NewtonLDA = NewtonLDA(DictArrayMatrix_Input=DictArrayMatrix_Data)
    ArrayMatrix_FisherW = Object_NewtonLDA.OriginalLDA()
    ArrayMatrix_LassoW = Object_NewtonLDA.LASSO_Soft(Lambda=Flt_Lambda)
    ArrayMatrix_LassoW2 = Object_NewtonLDA.LASSOtoLDA(Lambda=Flt_Lambda)
    Flt_FisherLDAVal = Object_NewtonLDA.ObjectiveFunction(ArrayMatrix_FisherW)
    Flt_LassoLDAVal = Object_NewtonLDA.ObjectiveFunction(ArrayMatrix_LassoW)

    print "Fisher LDA NO Constraint"
    print pd.DataFrame(ArrayMatrix_FisherW)
    print ""
    print "LASSO LDA with Lambda", Flt_Lambda
    print pd.DataFrame(ArrayMatrix_LassoW)
    print "Fisher LDA Value", Flt_FisherLDAVal
    print "Lasso LDA Value", Flt_LassoLDAVal

    # Transformed with W
    DictArrayMatrix_FLDATransformed = Object_NewtonLDA.FisherLDATransformation()
    DictArrayMatrix_LASSOTransformed = Object_NewtonLDA.LASSOLDATransformation(Lambda=Flt_Lambda)

    Object_NewtonLDA.PLOT(DictArrayMatrix=DictArrayMatrix_FLDATransformed, Str_Title=Str_FLDATitle)
    Object_NewtonLDA.PLOT(DictArrayMatrix=DictArrayMatrix_LASSOTransformed, Str_Title=Str_LassoTitle)
    plt.show()




