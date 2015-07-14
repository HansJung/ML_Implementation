# -*- coding: utf-8 -*-
'''
Goal : 
Author : Yonghan Jung, ISyE, KAIST 
Date : 150712 Kalman Filter for Noise Reduction
Comment 
- 

'''

''' Library '''
import numpy as np
import random
import matplotlib.pyplot as plt
''' Function or Class '''

class HansKalmanFilter:
    def __init__(self, Matrix_Test, Matrix_Train):
        def Compute_HatMatrix():
            try:
                return self.Matrix_Train * (self.Matrix_Train.T * self.Matrix_Train).I * self.Matrix_Train.T
            except:
                Matrix_Reg = np.matrix(np.eye(self.Int_NumTrain) * 1e-4)
                return self.Matrix_Train * (self.Matrix_Train.T * self.Matrix_Train + Matrix_Reg).I * self.Matrix_Train.T

        def Estimate_ProcessNoise():
            Matrix_TempTrain = self.Matrix_Train.T
            return np.matrix(np.diag(np.var(Matrix_TempTrain, axis=0)))

        # Each Column : Record. Column1, Column2...
        self.Matrix_Test = np.matrix(Matrix_Test)
        self.Matrix_Train = np.matrix(Matrix_Train)

        self.Int_SignalLen = len(self.Matrix_Train)
        self.Int_NumTrain = len(self.Matrix_Train.T)

        self.Matrix_HatMatrix = Compute_HatMatrix()
        self.Matrix_Q = Estimate_ProcessNoise()


    def KalmanPredict(self, Vector_MeanEst, Matrix_CovEst):
        Vector_MeanEst = np.reshape(Vector_MeanEst, (len(Vector_MeanEst),1))
        Vector_MeanPredict = Vector_MeanEst
        Matrix_CovPredict = Matrix_CovEst + self.Matrix_Q
        return (Vector_MeanPredict, Matrix_CovPredict)

    def Compute_MatrixR(self, Vector_Obs):
        Vector_Obs = np.reshape(Vector_Obs, (len(Vector_Obs),1))
        Vector_ObsHat = self.Matrix_HatMatrix * Vector_Obs
        Vector_Residual = Vector_Obs - Vector_ObsHat
        Flt_MeasureNoise = np.squeeze(np.asarray(Vector_Residual.T * Vector_Residual)) / float(self.Int_SignalLen - self.Int_NumTrain)
        Matrix_R = np.matrix(np.eye(self.Int_SignalLen) * Flt_MeasureNoise)
        return Matrix_R


    def KalmanUpdate(self, Vector_Obs, Vector_MeanPredict, Matrix_CovPredict, Vector_MeanEst):
        Matrix_R = self.Compute_MatrixR(Vector_Obs)
        Vector_Residual = Vector_Obs - Vector_MeanPredict
        Matrix_ResiaulCov = Matrix_CovPredict + Matrix_R
        try:
            Matrix_K = Matrix_CovPredict * (Matrix_ResiaulCov).I
        except:
            Matrix_Reg = np.matrix(np.eye(self.Int_SignalLen) * 1e-5)
            Matrix_K = Matrix_CovPredict * (Matrix_ResiaulCov + Matrix_Reg).I
        Vector_MeanEst = Vector_MeanEst + Matrix_K * Vector_Residual
        Matrix_CovEst = Matrix_CovPredict - Matrix_K * Matrix_ResiaulCov * Matrix_K.T

        return (Vector_MeanEst, Matrix_CovEst)

    def Execute_Kalman(self, Int_Niter, Vector_Obs):
        # Initializaiton
        # Vector_MeanEst = self.Matrix_HatMatrix * Vector_Obs
        Vector_MeanEst = np.reshape(np.ones(self.Int_SignalLen), (self.Int_SignalLen,1))
        Matrix_CovEst = np.matrix(np.eye(self.Int_SignalLen))

        for IntIdx in range(Int_Niter):
            print IntIdx
            (Vector_MeanPredict, Matrix_CovPredict) = self.KalmanPredict(Vector_MeanEst=Vector_MeanEst, Matrix_CovEst=Matrix_CovEst)
            (Vector_MeanEst, Matrix_CovEst) = self.KalmanUpdate(Vector_Obs=Vector_Obs, Matrix_CovPredict=Matrix_CovPredict, Vector_MeanPredict=Vector_MeanPredict, Vector_MeanEst = Vector_MeanEst)
        return (Vector_MeanEst, Matrix_CovEst)


def Generate_NoiseData(Int_SignalLength, Int_NumRecord, Matrix_Clean):
    Matrix_NoisyData = []
    for IntIdx in range(Int_NumRecord):
        Vector_Clean = np.array(Matrix_Clean.T[IntIdx])
        Noise = np.random.normal(20,40, Int_SignalLength)
        Vector_Obs = np.squeeze(np.asarray(np.reshape(Vector_Clean + Noise, (1, Int_SignalLength))))
        Matrix_NoisyData.append(Vector_Obs)
    Matrix_NoisyData = np.matrix(Matrix_NoisyData)
    Matrix_NoisyData = Matrix_NoisyData.T
    return Matrix_NoisyData

def Generate_CleanData(Int_SignalLength, Int_NumRecord):
    Matrix_CleanData = []

    # Initial Signal
    x = np.linspace(0,20,Int_SignalLength)
    y = 30*np.cos(2*x) + 20*np.sin(3*x)
    Vector_InitialSignal = y

    for IntIdx in range(Int_NumRecord):
        ProcessNoise = np.random.normal(0,1, Int_SignalLength)
        y += ProcessNoise
        Matrix_CleanData.append(Vector_InitialSignal)
    Matrix_CleanData = np.matrix(Matrix_CleanData)
    Matrix_CleanData = Matrix_CleanData.T
    return Matrix_CleanData

if __name__ == "__main__":
    # Generate Data
    Int_SigLen = 250
    Int_RecordNum = 40
    Int_KalmanIter = 1500

    Matrix_Clean = Generate_CleanData(Int_SignalLength= Int_SigLen, Int_NumRecord=Int_RecordNum)
    Matrix_Obs = Generate_NoiseData(Int_SignalLength= Int_SigLen, Int_NumRecord=Int_RecordNum,Matrix_Clean= Matrix_Clean)

    VectorObs = Matrix_Obs.T[30].T
    VectorClean = Matrix_Clean.T[30].T

    # Object
    Object_HansKalman = HansKalmanFilter(Matrix_Test=Matrix_Obs, Matrix_Train=Matrix_Clean)
    Vector_KalmanEst, _ = Object_HansKalman.Execute_Kalman(Int_KalmanIter,VectorObs)
    print np.sum(np.abs(Vector_KalmanEst - VectorClean))

    plt.plot(VectorObs,'b')
    plt.plot(VectorClean,'g')
    plt.plot(Vector_KalmanEst,'r')
    plt.show()


