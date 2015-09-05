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
from sklearn.linear_model import Lasso
from sklearn.datasets import load_boston
from TwoClassFisherLDA import TwoClassFisherLDA
import matplotlib.pyplot as plt

''' Function or Class '''

def Generate_Data(Mu1, Mu2, Dim, ObsNum_Class1, ObsNum_Class2):
    Mu1 = [Mu1] * Dim
    Mu2 = [Mu2] * Dim
    Cov = np.eye(Dim)
    # Dim * 100 Mat
    Data1 = np.random.multivariate_normal(Mu1, Cov, ObsNum_Class1)
    Data2 = np.random.multivariate_normal(Mu2, Cov, ObsNum_Class2)

    Data = np.concatenate((Data1, Data2), axis=0)

    Dict_Data = dict()
    Dict_Data[0] = Data1
    Dict_Data[1] = Data2
    return Dict_Data, Data


if __name__ == "__main__":
    ObsNum_Class1 = 100
    ObsNum_Class2 = 100
    ObsNum = ObsNum_Class1 + ObsNum_Class2
    ClassNumber = 2

    Mu1 = 1
    Mu2 = -1
    Dim = 30



    # Data_Boston = load_boston()
    # X = Data_Boston['data']
    # y = Data_Boston['target']
    # print y

    Flt_Lambda = 0.3

    Dict_Data, Array_Data = Generate_Data(Mu1, Mu2, Dim, ObsNum_Class1, ObsNum_Class2)
    Data1 = Dict_Data[0]
    Data2 = Dict_Data[1]

    ObjLDA = TwoClassFisherLDA(Dict_Data)
    W = ObjLDA.ConstructW()


    Y = np.zeros((ObsNum, ClassNumber))
    for idx in range(len(Y)):
        if idx < ObsNum_Class1:
            Y[idx][0] = 1
        else:
            Y[idx][1] = 1

    # 각 대각은 각기 class가 차지하는 비율
    D = np.dot(np.transpose(Y), Y) / float(ObsNum)

    # Initialize Q
    Q = np.ones((ClassNumber,1))

    # Optimal score (Theta)
    # InitialTheta = np.ones((2,1))
    InitialTheta = np.array([2,5])
    I = np.eye(ClassNumber)
    Theta = np.dot(I - np.dot(np.dot(Q, np.transpose(Q)), D ), InitialTheta)
    Theta /= np.sqrt(np.dot(np.dot(np.transpose(Theta), D), Theta))
    # print np.dot(np.dot(np.transpose(Theta), D), Theta)
    for idx in range(50):
        NewResp = np.dot(Y, Theta)
        lasso = Lasso(alpha=Flt_Lambda)
        #
        # # Compute Coefficient
        B = lasso.fit(X=Array_Data, y= NewResp).coef_
        # print B
        #
        # New OptScore
        Part1 = I - np.dot(np.dot(Q, np.transpose(Q)),D)
        Part2 = np.dot(Part1, np.linalg.inv(D))
        Part3 = np.dot(Part2, np.transpose(Y))
        WaveTheta = np.dot(np.dot(Part3, Array_Data), B)
        # print WaveTheta
        Theta = WaveTheta / np.sqrt(np.dot(np.dot(np.transpose(WaveTheta),D),WaveTheta))
        # print B
        # print idx, B

    W = np.squeeze(np.transpose(W))
    print W
    print "-" * 30
    print B
    Data1_Transform = np.dot(Data1, W)
    Data2_Transform = np.dot(Data2, W)

    Data1_Transform_B = np.dot(Data1, B)
    Data2_Transform_B = np.dot(Data2, B)

    plt.figure()
    plt.title("Coefficients")
    plt.grid()
    plt.plot(W * 10, 'bo', label="LDA")
    plt.plot(B, 'ro', label="Optimal")
    plt.legend()

    plt.figure()
    plt.title("Original Data ")
    for data1 in Data1:
        for idx, data1_elem in enumerate(data1):
            plt.plot(idx, data1_elem, 'bo')
    for data2 in Data2:
        for idx, data2_elem in enumerate(data2):
            plt.plot(idx, data2_elem, 'ro')
    plt.grid()
    plt.legend()



    plt.figure()
    plt.title("Optimal Score Transformed")
    plt.grid()
    plt.plot(Data1_Transform_B,'bo', label="Class1")
    plt.plot(Data2_Transform_B,'ro', label="Class2")
    plt.legend()

    plt.figure()
    plt.title("LDA Transformed")
    plt.grid()
    plt.plot(Data1_Transform,'bo', label="Class1")
    plt.plot(Data2_Transform,'ro', label="Class2")


    plt.show()









