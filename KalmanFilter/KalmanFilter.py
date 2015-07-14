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
import pandas as pd
import matplotlib.pyplot as plt
from random import *

''' Function or Class '''


class HansKalman:
    def __init__(self):
        self.SeedNum = 100
        self.Int_SignalLength = 100
        self.Int_TrainFeatureNum = 30
        self.Int_TestFeatureNum = 50

    def KalmanFilter(self):
        ### Initial ###
        Array_InitialMu = np.ones(self.Int_SignalLength)
        Array_Mu = np.reshape(Array_InitialMu, (self.Int_SignalLength,1))
        Matrix_Sigma = np.eye(self.Int_SignalLength)
        Matrix_MeasurementNoise = np.eye(self.Int_SignalLength)
        Matrix_ProcessNoise = np.eye(self.Int_SignalLength)

        Matrix_Id = np.matrix(np.eye(self.Int_SignalLength))

        ### Compute HatMatrix ###
        ArrayMatrix_TrainData, List_TrainLabel = self.Generate_TrainingData()
        ArrayMatrix_TrainData = np.matrix(ArrayMatrix_TrainData)
        try:
            ArrayMatrix_HatMatrix = ArrayMatrix_TrainData * (ArrayMatrix_TrainData.T * ArrayMatrix_TrainData).I * ArrayMatrix_TrainData.T
        except:
            row, col = ArrayMatrix_TrainData.shape
            Matrix_Reg = np.matrix(np.eye(col) * 1e-5)
            ArrayMatrix_HatMatrix = ArrayMatrix_TrainData * (ArrayMatrix_TrainData.T * ArrayMatrix_TrainData + Matrix_Reg).I * ArrayMatrix_TrainData.T

        ### Load Test Signal ###
        Matrix_ObsSignal, Label = self.Generate_ObservationData()

        for EachObs in Matrix_ObsSignal.T:
            EachObs = np.reshape(EachObs, (self.Int_SignalLength, 1))
            Residual = EachObs - ArrayMatrix_HatMatrix * EachObs
            MeasurementNoise = np.squeeze(np.asarray(Residual.T * Residual)) / float(self.Int_SignalLength - self.Int_TrainFeatureNum)
            Matrix_MeasurementNoise = np.matrix(np.eye(self.Int_SignalLength) * MeasurementNoise)

            Matrix_KalmanGain = (Matrix_Sigma + Matrix_MeasurementNoise) * ((Matrix_Sigma + Matrix_MeasurementNoise + Matrix_ProcessNoise).I)
            Matrix_Sigma = (Matrix_Id - Matrix_KalmanGain) * (Matrix_Sigma + Matrix_MeasurementNoise)
            Array_Mu += Matrix_KalmanGain * (EachObs - Array_Mu)
        return Array_Mu



    ''' Data Generation '''
    # Col : Each Record
    # Row : Each Signal Feature
    def Generate_ObservationData(self):
        def Generate_CleanDataLabel0():
            x = np.linspace(0,10,self.Int_SignalLength)
            y = np.cos(2*x)
            z = np.sin(3*x)
            return y+z

        def Generate_CleanDataLabel1():
            x = np.linspace(0,10,self.Int_SignalLength)
            y = np.cos(2*x)
            z = np.sin(3*x)
            return y+z

        def Generate_Noise():
            np.random.seed(self.SeedNum)
            Noise = np.random.normal(0,1,self.Int_SignalLength)
            return Noise
        np.random.seed(self.SeedNum)
        randBinList = lambda n: [randint(0,1) for b in range(1,n+1)]
        DataNum = self.Int_TestFeatureNum

        ObservationMatrix = []
        Label = randBinList(DataNum)

        for IntIter in range(DataNum):
            if Label[IntIter] == 0:
                Array_PureSignal = np.squeeze(np.asarray(Generate_CleanDataLabel0()))
            elif Label[IntIter] == 1:
                Array_PureSignal = np.squeeze(np.asarray(Generate_CleanDataLabel1()))
            Array_Noise = np.squeeze(np.array(Generate_Noise()))
            Array_EachObs = Array_PureSignal + Array_Noise
            ObservationMatrix.append(Array_EachObs)
        ObservationMatrix = np.asarray(ObservationMatrix)
        ObservationMatrix = ObservationMatrix.T
        return ObservationMatrix, Label

    # Col : Each Record
    # Row : Each Signal Feature
    def Generate_TrainingData(self):
        def Generate_CleanDataLabel0():
            x = np.linspace(0,10,self.Int_SignalLength)
            y = np.cos(2*x)
            z = np.sin(3*x)
            return y+z

        def Generate_CleanDataLabel1():
            x = np.linspace(0,10,self.Int_SignalLength)
            y = np.cos(2*x)
            z = np.sin(3*x)
            return y+z

        def Generate_Noise():
            np.random.seed(self.SeedNum)
            Noise = np.random.normal(0,1,self.Int_SignalLength)
            return Noise
        np.random.seed(self.SeedNum)
        randBinList = lambda n: [randint(0,1) for b in range(1,n+1)]
        DataNum = self.Int_TrainFeatureNum

        ObservationMatrix = []
        Label = randBinList(DataNum)

        for IntIter in range(DataNum):
            if Label[IntIter] == 0:
                Array_PureSignal = np.squeeze(np.asarray(Generate_CleanDataLabel0()))
            elif Label[IntIter] == 1:
                Array_PureSignal = np.squeeze(np.asarray(Generate_CleanDataLabel1()))
            ObservationMatrix.append(Array_PureSignal)
        ObservationMatrix = np.asarray(ObservationMatrix)
        ObservationMatrix = ObservationMatrix.T
        return ObservationMatrix, Label

    ''' Measurement Error '''
    def RegressionLSE(self, Array_SigleObs):
        Int_Dim = len(Array_SigleObs)
        Array_SigleObs = np.matrix(np.reshape(Array_SigleObs,(Int_Dim,1)))

        ArrayMatrix_TrainData, List_TrainLabel = self.Generate_TrainingData()
        ArrayMatrix_TrainData = np.matrix(ArrayMatrix_TrainData)
        try:
            ArrayMatrix_HatMatrix = ArrayMatrix_TrainData * (ArrayMatrix_TrainData.T * ArrayMatrix_TrainData).I * ArrayMatrix_TrainData.T
        except:
            row, col = ArrayMatrix_TrainData.shape
            Matrix_Reg = np.matrix(np.eye(col) * 1e-5)
            ArrayMatrix_HatMatrix = ArrayMatrix_TrainData * (ArrayMatrix_TrainData.T * ArrayMatrix_TrainData + Matrix_Reg).I * ArrayMatrix_TrainData.T

        Array_LSESignal = ArrayMatrix_HatMatrix * Array_SigleObs
        Residual = Array_SigleObs - Array_LSESignal

        return np.squeeze(np.asarray(Array_LSESignal)), Residual





if __name__ == "__main__":
    Object_HansKalman = HansKalman()

    Array_TrainingSignal, List_TrainLabel = Object_HansKalman.Generate_TrainingData()
    Array_ObsSignal, List_ObsLabel = Object_HansKalman.Generate_ObservationData()
    Array_SigleObs = Array_ObsSignal.T[49]
    Array_CleanObs = Array_TrainingSignal.T[0]
    Array_LSESigleObs, Residual = Object_HansKalman.RegressionLSE(Array_SigleObs)
    Array_Mu = Object_HansKalman.KalmanFilter()

    Int_Dim = len(Residual)
    Int_FeatureDim = Object_HansKalman.Int_TrainFeatureNum

    print np.squeeze(np.array(Residual.T * Residual)) / float(Int_Dim - Int_FeatureDim)

    print Array_Mu

    plt.grid()
    plt.plot(Array_Mu,'b')
    plt.plot(Array_SigleObs,'r')
    plt.plot(Array_CleanObs,'g')
    plt.show()
