# -*- coding: utf-8 -*-
'''
Goal : To simulate the kNN
Author : Yonghan Jung, ISyE, KAIST 
Date : 150503
Comment 
- 

'''

''' Library '''
import numpy as np
import operator

''' Function or Class '''

def CreateData():
    Data = np.array([1,1.1],[1,1],[0,0],[0,0.1], dtype='float32')
    Label = [0,0,1,1] # List
    return Data, Label

if __name__ == "__main__":
    Data, Label = CreateData()
