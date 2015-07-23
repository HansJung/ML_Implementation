# -*- coding: utf-8 -*-
'''
Goal : LASSO Practice
Author : Yonghan Jung, ISyE, KAIST 
Date : 150722
Comment 
- 

'''

''' Library '''
import numpy as np
from sklearn.datasets import load_boston
import matplotlib.pyplot as plt
from sklearn.linear_model import Lasso

''' Function or Class '''


class LASSO_Practice:
    def __init__(self):
        return None

    def Construct_HatMatrix(self, Mat_X):
        return np.dot(Mat_X, np.dot(np.linalg.inv(np.dot(np.transpose(Mat_X), Mat_X)), np.transpose(Mat_X)))

    def Construc_Coefficient(self, Mat_X, Mat_y):
        return np.dot(np.dot(np.linalg.inv(np.dot(np.transpose(Mat_X), Mat_X)), np.transpose(Mat_X)), Mat_y)

    def Objective_Function(self, Mat_Coef, Mat_X, Mat_y, Flt_Lambda):
        N = len(Mat_y)
        return 1/float(N) * np.linalg.norm(Mat_y - np.dot(Mat_X, Mat_Coef),2 ) + Flt_Lambda * np.linalg.norm(Mat_Coef, 1)


if __name__ == "__main__":
    Data_Boston = load_boston()
    Object_Lasso = LASSO_Practice()

    Mat_X = Data_Boston['data']
    Mat_y = Data_Boston['target']

    Mat_Hat = Object_Lasso.Construct_HatMatrix(Mat_X)
    Mat_Coef = Object_Lasso.Construc_Coefficient(Mat_X, Mat_y)
    Mat_yhat = np.dot(Mat_Hat, Mat_y)

    Flt_Lambda = 1
    lasso = Lasso(alpha=Flt_Lambda)
    Array_LassoCoef = lasso.fit(Mat_X, Mat_y).coef_
    Array_TrueCoef = Mat_Coef

    for a,b in zip(Array_LassoCoef, Array_TrueCoef):
        print np.round(a,2), np.round(b,2)