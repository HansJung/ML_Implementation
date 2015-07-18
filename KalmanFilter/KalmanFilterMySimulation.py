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
''' Function or Class '''

class GenerateData:
    def __init__(self):
        pass
    def GenerateTwoClassSignal(self):
        Int_SignalLength = 256
        Array_TimeDomain = np.linspace(0,30,Int_SignalLength)
        Array_ClassOneSignal = 100*Array_TimeDomain + 10*np.sin(20*Array_TimeDomain)
        Array_ClassTwoSignal = -100 * Array_TimeDomain - 10* np.cos(20*Array_TimeDomain)

        Array_Noise = np.random.normal(0,30, Int_SignalLength)
        Array_ClassOneSignal += Array_Noise
        Array_ClassTwoSignal += Array_Noise
        return Array_ClassOneSignal, Array_ClassTwoSignal

    def GenerateProcessSignal(self, Int_NumClassOne, Int_NumClassTwo):
        Array_ClassOneSignal, Array_ClassTwoSignal = self.GenerateTwoClassSignal()
        Int_SignalLength = len(Array_ClassOneSignal)
        Matrix_RowElem_ColSig = list()
        Dict_Class_Sig = dict()
        Dict_Class_Sig[1] = list()
        Dict_Class_Sig[2] = list()

        for Idx in range(Int_NumClassOne):
            Array_ProcessNoise = np.random.normal(0,5, Int_SignalLength)
            Array_ClassOneNoisy = Array_ClassOneSignal + Array_ProcessNoise
            Matrix_RowElem_ColSig.append(Array_ClassOneNoisy)
            Dict_Class_Sig[1].append(Array_ClassOneNoisy)

        for Idx in range(Int_NumClassTwo):
            Array_ProcessNoise = np.random.normal(0,5, Int_SignalLength)
            Array_ClassTwoNoisy = Array_ClassTwoSignal + Array_ProcessNoise
            Matrix_RowElem_ColSig.append(Array_ClassTwoNoisy)
            Dict_Class_Sig[2].append(Array_ClassTwoNoisy)

        Matrix_RowElem_ColSig = np.array(Matrix_RowElem_ColSig)
        Matrix_RowElem_ColSig = np.transpose(Matrix_RowElem_ColSig)
        Dict_Class_Sig[1] = np.array(Dict_Class_Sig[1])
        Dict_Class_Sig[2] = np.array(Dict_Class_Sig[2])
        return Matrix_RowElem_ColSig, Dict_Class_Sig


    def GenerateMeasurementSignal(self, Int_NumClassOne, Int_NumClassTwo):
        Array_ClassOneSignal, Array_ClassTwoSignal = self.GenerateTwoClassSignal()
        Int_SignalLength = len(Array_ClassOneSignal)
        Matrix_RowElem_ColSig = list() # For not sparsity
        Dict_Class_Sig = dict()
        Dict_Class_Sig[1] = list()
        Dict_Class_Sig[2] = list()

        for Idx in range(Int_NumClassOne):
            Array_ProcessNoise = np.random.normal(0,5, Int_SignalLength)
            Array_MeasureNoise = np.random.normal(0,10, Int_SignalLength)
            Array_ClassOneNoisy = Array_ClassOneSignal + Array_ProcessNoise + Array_MeasureNoise
            Matrix_RowElem_ColSig.append(Array_ClassOneNoisy)
            Dict_Class_Sig[1].append(Array_ClassOneNoisy)

        for Idx in range(Int_NumClassTwo):
            Array_ProcessNoise = np.random.normal(0,5, Int_SignalLength)
            Array_MeasureNoise = np.random.normal(0,10, Int_SignalLength)
            Array_ClassTwoNoisy = Array_ClassTwoSignal + Array_ProcessNoise+ Array_MeasureNoise
            Matrix_RowElem_ColSig.append(Array_ClassTwoNoisy)
            Dict_Class_Sig[2].append(Array_ClassTwoNoisy)
        Matrix_RowElem_ColSig = np.array(Matrix_RowElem_ColSig)
        Matrix_RowElem_ColSig = np.transpose(Matrix_RowElem_ColSig)
        Dict_Class_Sig[1] = np.array(Dict_Class_Sig[1])
        Dict_Class_Sig[2] = np.array(Dict_Class_Sig[2])
        return Matrix_RowElem_ColSig, Dict_Class_Sig




class KalmanSimulator:
    def __init__(self):
        return None

    ### Generate Class


if __name__ == "__main__":

    Int_NumClassOne = 257
    Int_NumClassTwo = 1
    ExampleNum = 257

    Object_Data = GenerateData()

    Matrix_Noisy_RowElem_ColSig, _ = Object_Data.GenerateMeasurementSignal(Int_NumClassOne=Int_NumClassOne, Int_NumClassTwo=Int_NumClassTwo)
    Matrix_RowElem_ColSig, _ = Object_Data.GenerateProcessSignal(Int_NumClassOne=Int_NumClassOne, Int_NumClassTwo=Int_NumClassTwo)
    Matrix_Hat = np.dot(Matrix_RowElem_ColSig, np.dot(np.linalg.inv(np.dot(np.transpose(Matrix_RowElem_ColSig), Matrix_RowElem_ColSig)), np.transpose(Matrix_RowElem_ColSig) ))

    Array_Example_CleanSignal = np.transpose(Matrix_RowElem_ColSig)[ExampleNum]
    Array_Example_NoisySignal = np.transpose(Matrix_Noisy_RowElem_ColSig)[ExampleNum]
    Array_HatSignal = np.dot(Matrix_Hat, Array_Example_NoisySignal)

    plt.figure()
    plt.grid()
    plt.plot(Array_Example_CleanSignal,'b', label = 'Clean')
    plt.plot(Array_Example_NoisySignal,'r', label = "Raw")
    plt.plot(Array_HatSignal,'g', label="Est.")
    plt.show()