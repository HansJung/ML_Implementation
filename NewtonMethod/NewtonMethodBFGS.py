# -*- coding: utf-8 -*-
'''
Goal : 
Author : Yonghan Jung, ISyE, KAIST 
Date : 150709 BFGS
Comment 
- 

'''

''' Library '''
import numpy as np
import matplotlib.pyplot as plt
from Fisher_Score_Computation import Fisher_Score
import pandas as pd
import pprint
''' Function or Class '''


class BFGStoLDA:
    def __init__(self, Flt_Alpha, Flt_Beta, DictArrayMatrix_Input, Lambda, Threshold, LDAON):
        self.Flt_Alpha = Flt_Alpha
        self.Flt_Beta = Flt_Beta
        # Key : 0, 1
        # For each key,
        self.DictArrayMatrix_Input = DictArrayMatrix_Input
        self.DictArrayMatrix_Concatenate = np.concatenate([DictArrayMatrix_Input[0], DictArrayMatrix_Input[1]])
        self.Int_Dim = len(self.DictArrayMatrix_Concatenate[0])
        self.Array_Class1Mean = np.mean(DictArrayMatrix_Input[0],axis=0)
        self.Array_Class2Mean = np.mean(DictArrayMatrix_Input[1],axis=0)
        self.TotalMean = np.mean(self.DictArrayMatrix_Concatenate, axis = 0)
        self.Lambda = Lambda
        self.Threshold = Threshold
        self.LDAON = LDAON


    ######### Compute LDA Parameter ########
    ### BetweenScatter Matrix ###

    def BetweenScatter(self):
        ArrayColumn_Class1Mu = np.reshape(self.Array_Class1Mean - self.TotalMean, (self.Int_Dim,1))
        ArrayRow_Class1Mu = np.reshape(self.Array_Class1Mean - self.TotalMean, (1,self.Int_Dim))
        ArrayMatrix_Class1 = ArrayColumn_Class1Mu * ArrayRow_Class1Mu

        ArrayColumn_Class2Mu = np.reshape(self.Array_Class2Mean - self.TotalMean, (self.Int_Dim,1))
        ArrayRow_Class2Mu = np.reshape(self.Array_Class2Mean - self.TotalMean, (1,self.Int_Dim))
        ArrayMatrix_Class2 = ArrayColumn_Class2Mu * ArrayRow_Class2Mu

        return np.matrix(ArrayMatrix_Class1 + ArrayMatrix_Class2)

    ### Within Matrix with small Regularization ###
    def WithinScatter(self):
        ArrayMatrix_Class1Scatter = np.zeros((self.Int_Dim,self.Int_Dim))
        Flt_SmallNumRegularizer = 1e-8
        Matrix_Regularization = np.eye(self.Int_Dim) * Flt_SmallNumRegularizer
        for Array_Class1Data in self.DictArrayMatrix_Input[0]:
            ArrayColumn_DiffClass1 = np.reshape(Array_Class1Data - self.Array_Class1Mean,(self.Int_Dim,1))
            ArrayRow_DiffClass1 = np.reshape(Array_Class1Data - self.Array_Class1Mean,(1,self.Int_Dim))
            ArrayMatrix_Class1Scatter += ArrayColumn_DiffClass1 * ArrayRow_DiffClass1

        for Array_Class2Data in self.DictArrayMatrix_Input[1]:
            ArrayColumn_DiffClass2 = np.reshape(Array_Class2Data - self.Array_Class2Mean,(self.Int_Dim,1))
            ArrayRow_DiffClass2 = np.reshape(Array_Class2Data - self.Array_Class2Mean,(1,self.Int_Dim))
            ArrayMatrix_Class1Scatter += ArrayColumn_DiffClass2 * ArrayRow_DiffClass2
        return np.matrix(ArrayMatrix_Class1Scatter) + Matrix_Regularization

    # W: LDA Operator, Matrix (dim by dim)
    def LDAValue(self, W):
        S_B = self.BetweenScatter()
        S_W = self.WithinScatter()
        MatrixRegularization = 1e-6 * np.eye(self.Int_Dim)
        Matrix_Numerator = W.T * S_B * W
        try :
            Matrix_Denominator = (W.T * S_W * W ).I
        except :
            Matrix_Denominator = (W.T * S_W * W + MatrixRegularization ).I
        return -1 * np.trace(Matrix_Numerator * (Matrix_Denominator))

    def Vectorization(self,MatrixA):
        # A : n by d matrix.
        # return : nd by 1
        MatrixA = np.squeeze(np.asarray(MatrixA))
        IntRow, IntCol = MatrixA.shape
        return np.reshape(MatrixA, (IntRow * IntCol, 1))

    def Inverse_Vectorization(self,VectorA, IntRow, IntCol):
        return np.reshape(VectorA, (IntRow,IntCol)).T

    def Delta_LDA(self,W):
        S_B = self.BetweenScatter()
        S_W = self.WithinScatter()
        MatrixRegularization = 1e-6 * np.eye(self.Int_Dim)
        try:
            return S_W * W * ((W.T* S_W * W ).I) * (W.T * S_B * W) * ((W.T * S_W * W ).I) - 2 * S_B *W * ((W.T * S_W *W).I)
        except:
            return S_W * W * ((W.T* S_W * W + MatrixRegularization ).I) * (W.T * S_B * W) * ((W.T * S_W * W + MatrixRegularization).I) - 2 * S_B *W * ((W.T * S_W *W + MatrixRegularization).I)

    def LASSO_FLDA(self,OriginalW):
        def Sgn(AnyVal):
            if AnyVal > 0 :
                return 1
            elif AnyVal < 0:
                return -1
            elif AnyVal == 0:
                return 0

        def SoftThreshold(VectorA, LambdaVector):
            MaxIdx = np.argmax(np.abs(VectorA))
            if np.max((np.abs(VectorA) - LambdaVector)) > 0:
                NewVector = Sgn(VectorA[MaxIdx]) * (np.abs(VectorA) - LambdaVector)
                return NewVector
            else:
                print "ho"
                NewVector = np.zeros(self.Int_Dim)
                return NewVector

        OriginalW = self.OriginalLDA()
        DeltaW = self.Delta_LDA(OriginalW)
        NewW = []

        for Int_RowIdx in range(self.Int_Dim):
            Wi = np.squeeze(np.asarray(OriginalW[Int_RowIdx]))
            LambdaVector = np.ones(self.Int_Dim) * self.Lambda
            Wi = SoftThreshold(Wi, LambdaVector)
            Wi = np.squeeze(np.asarray(Wi))
            NewW.append(Wi)
        NewW = np.array(NewW)
        return NewW



    def ComputingNewDelta(self, W, Delta):
        def Sgn(AnyInt):
            if AnyInt > 0:
                return 1
            elif AnyInt < 0:
                return -1
            elif AnyInt == 0:
                return 0

        # Delta is a matrix (d by d)
        NewDelta = []
        for Int_RowIdx in range(self.Int_Dim):
            # For each row of Wi and Deltai
            Wi = np.squeeze(np.asarray(W[Int_RowIdx]))
            Deltai = np.squeeze(np.asarray(Delta[Int_RowIdx]))

            # Size of Wi and Deltai
            InfNorm_Wi = np.max(np.abs(Wi))
            MaxWiIdx = np.argmax(np.abs(Wi))
            InfNorm_Deltai = np.max(np.abs(Deltai))
            MaxDeltaiIdx = np.argmax(np.abs(Deltai))

            if InfNorm_Wi > self.Threshold:
                NewDeltai = Deltai + self.Lambda * np.ones(self.Int_Dim) * Sgn(Wi[MaxWiIdx])
            elif InfNorm_Wi <= self.Threshold and Deltai[MaxDeltaiIdx] < - self.Lambda:
                NewDeltai = Deltai + self.Lambda * np.ones(self.Int_Dim)
            elif InfNorm_Wi <= self.Threshold and Deltai[MaxDeltaiIdx] > self.Lambda:
                NewDeltai = Deltai + (-1) * self.Lambda * np.ones(self.Int_Dim)
            elif InfNorm_Wi <= self.Threshold and Deltai[MaxDeltaiIdx] > - self.Lambda and Deltai[MaxDeltaiIdx] < self.Lambda:
                NewDeltai = np.zeros(self.Int_Dim)
            NewDelta.append(np.squeeze(np.asarray(NewDeltai)))
        NewDelta = np.asarray(NewDelta)

        return NewDelta


    def ApproxHessianInverse(self, C, s, y):
        IdMat = np.eye(self.Int_Dim)
        RegularizedMatrix = np.eye(self.Int_Dim) * 1e-6
        return (IdMat - (s * y.T) * (y.T * s + RegularizedMatrix).I) * C * (IdMat - (y*s.T * (y.T * s + RegularizedMatrix).I)) + s*s.T*((y.T*s+RegularizedMatrix).I)

    def ComputeNewW(self, PrevW, W):
        NewW = []
        for Int_Row in range(self.Int_Dim):
            PrevWi = np.squeeze(np.asarray(PrevW[Int_Row]))
            Wi = np.squeeze(np.asarray(W[Int_Row]))
            Wi_MaxIdx = np.argmax(np.abs(Wi))
            PrevWi_MaxIdx = np.argmax(np.abs(PrevWi))

            if Wi[Wi_MaxIdx] * PrevWi[PrevWi_MaxIdx] < 0 :
                NewWi = np.zeros(self.Int_Dim)
            else:
                NewWi = Wi
            NewW.append(NewWi)
        NewW = np.array(NewW)
        return NewW




    def BFGSAlgorithm(self):
        ## Initialization (k=0)
        W = np.eye(self.Int_Dim)
        # W = np.ones((self.Int_Dim,self.Int_Dim)) + np.eye(self.Int_Dim)
        Int_Iter = 0
        C = np.eye(self.Int_Dim )
        DeltaLDA = self.Delta_LDA(W)
        while True:
            Int_Iter += 1
            Prev_W = W
            Prev_DeltaLDA = DeltaLDA
            # print "hoho",(np.dot(C,DeltaLDA)).shape
            # W = W - 0.2*np.array(np.matrix(C) * np.matrix(DeltaLDA))
            NewDelta = self.ComputingNewDelta(W,DeltaLDA)
            print Int_Iter, "th Delta"
            print pd.DataFrame(DeltaLDA)
            print Int_Iter, "th NewDelta"
            print pd.DataFrame(NewDelta)
            # W = W - DeltaLDA
            W = W - NewDelta
            if self.LDAON == True:
                W = self.ComputeNewW(PrevW=Prev_W, W = W )
            else:
                pass
            DeltaLDA = self.Delta_LDA(W)
            # print "DELTA", DeltaLDA
            s = W - Prev_W
            y = DeltaLDA - Prev_DeltaLDA
            # C = self.ApproxHessianInverse(C,s,y)
            # print Int_Iter, "th Hessian Matrix"
            # print pd.DataFrame(C)
            print Int_Iter, "th iter's W LDA VAL", self.LDAValue(W)
            print Int_Iter, "th W"
            print pd.DataFrame(W)
            Distance = np.trace((W - Prev_W) * (W - Prev_W).T)
            print "Distance", Distance
            print "-" * 100
            if Distance < 1e-4 or Int_Iter > 100:
            # if Int_Iter > 100:
                print "WOWWOWWOWWOW", Int_Iter, "DONE"
                break
        return W

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
        return SQRT_S_W * EigMat









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
    ArrayMatrix_Cov2 = np.eye(Int_Dim)
    DictArrayMatrix[1] = np.random.multivariate_normal(Array_Mu2, ArrayMatrix_Cov2, Int_DataNum/4)

    return DictArrayMatrix


if __name__ == "__main__":
    MatrixA = np.array([[1,2,3],[4,5,6], [7,8,9]])


    Int_Seed = 1
    Int_Mu1 = 1
    Int_Mu2 = -1
    Int_Dim = 10
    Int_DataNum = 100
    Threshold = 0.000

    Lambda = 8

    LDAON = True
    PLOT = True
    # LDAON = False
    PLOT = False


    DictArrayMatrix_Data = GeneratingSimulationData(Int_Seed = Int_Seed, Int_Mu1 = Int_Mu1, Int_Mu2 = Int_Mu2, Int_Dim = Int_Dim, Int_DataNum=Int_DataNum)
    DictArrayMatrix_TotalData = np.concatenate([DictArrayMatrix_Data[0], DictArrayMatrix_Data[1]])
    Array_Class1Mean = np.mean(DictArrayMatrix_Data[0], axis=0)
    Array_Class2Mean =  np.mean(DictArrayMatrix_Data[1], axis=0)
    Array_ClassTotalMean = np.mean(DictArrayMatrix_TotalData,axis=0)

    Obj_BFGStoLDA = BFGStoLDA(0.1,0.1,DictArrayMatrix_Data, Lambda=Lambda, Threshold=Threshold, LDAON = LDAON)

    OriginalW = Obj_BFGStoLDA.OriginalLDA()
    MyW = Obj_BFGStoLDA.BFGSAlgorithm()
    Delta = Obj_BFGStoLDA.Delta_LDA(OriginalW)
    LASSOW = Obj_BFGStoLDA.LASSO_FLDA(OriginalW)

    print "FLDA W"
    print pd.DataFrame(OriginalW)
    print "LASSO W"
    print pd.DataFrame(LASSOW)
    print "MyW W "
    print pd.DataFrame(MyW)
    print ""
    print "ANSWER LDA VAL", Obj_BFGStoLDA.LDAValue(OriginalW)
    print "MyW LDA VAL", Obj_BFGStoLDA.LDAValue(MyW)
    print "LASSO LDA VAL", Obj_BFGStoLDA.LDAValue(LASSOW)
    print ""

    #
    TransformedClass0 = np.matrix(DictArrayMatrix_Data[0]) * LASSOW.T
    TransformedClass1 = np.matrix(DictArrayMatrix_Data[1]) * LASSOW.T
    # print "OriginalData"
    # print pd.DataFrame(DictArrayMatrix_Data[0])
    # print "Transformed"
    # print pd.DataFrame(TransformedClass0)

    NewTransformedClass0 = np.matrix(DictArrayMatrix_Data[0]) * OriginalW
    NewTransformedClass1 = np.matrix(DictArrayMatrix_Data[1]) * OriginalW

    DictArrayMatrix_TransformedData = dict()
    DictArrayMatrix_TransformedData[0] = np.squeeze(np.asarray(TransformedClass0))
    DictArrayMatrix_TransformedData[1] = np.squeeze(np.asarray(TransformedClass1))

    DictArrayMatrix_NewTransformedData = dict()
    DictArrayMatrix_NewTransformedData[0] = np.squeeze(np.asarray(NewTransformedClass0))
    DictArrayMatrix_NewTransformedData[1] = np.squeeze(np.asarray(NewTransformedClass1))

    # print DictArrayMatrix_TransformedData[0][0]

    Obj_Fisher_Original = Fisher_Score(DictArrayMatrix_Data)
    Obj_Fisher_Transformed = Fisher_Score(DictArrayMatrix_TransformedData)
    Obj_Fisher_Answer = Fisher_Score(DictArrayMatrix_NewTransformedData)

    print "Before Transformed"
    for idx, val in enumerate(Obj_Fisher_Original.Fisher_Score()):
        print idx, val
    print ""
    print "After Transformed"
    for idx, val in enumerate(Obj_Fisher_Transformed.Fisher_Score()):
        print idx, val
    print ""
    print "After True LDA Transformed"
    for idx, val in enumerate(Obj_Fisher_Answer.Fisher_Score()):
        print idx, val

    if PLOT == True:
        plt.figure()
        plt.grid()
        plt.title("BEFORE")



        for Class1Data in DictArrayMatrix_Data[0]:
            for idx, eachdata in enumerate(Class1Data):
                plt.plot(idx, eachdata, 'bo')

        for Class2Data in DictArrayMatrix_Data[1]:
            for idx, eachdata in enumerate(Class2Data):
                plt.plot(idx, eachdata, 'ro')

        plt.figure()
        plt.grid()
        plt.title("AFTER")

        for Class1Data in TransformedClass0:
            Class1Data = np.squeeze(np.asarray(Class1Data))
            for idx, eachdata in enumerate(Class1Data):
                plt.plot(idx, eachdata, 'bo')

        for Class2Data in TransformedClass1:
            Class2Data = np.squeeze(np.asarray(Class2Data))
            for idx, eachdata in enumerate(Class2Data):
                plt.plot(idx, eachdata, 'ro')

        plt.figure()
        plt.grid()
        plt.title("Answer")

        for Class1Data in NewTransformedClass0:
            Class1Data = np.squeeze(np.asarray(Class1Data))
            for idx, eachdata in enumerate(Class1Data):
                plt.plot(idx, eachdata, 'bo')

        for Class2Data in NewTransformedClass1:
            Class2Data = np.squeeze(np.asarray(Class2Data))
            for idx, eachdata in enumerate(Class2Data):
                plt.plot(idx, eachdata, 'ro')


        plt.show()