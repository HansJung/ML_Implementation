# -*- coding: utf-8 -*-
'''
Goal : Test Wavelet decomposition
Author : Yonghan Jung, IE, KAIST 
Date : 150501
Comment 
- 
'''

''' Library '''
import pywt
import numpy as np
import Wavelet
''' Function or Class '''

if __name__ == "__main__":
    A = np.array([1,2,3,4,5,6,7,8], float)
    db4 = pywt.Wavelet('db4')
    print pywt.wavedec(data = A, mode = 'per', wavelet = db4, level = 1)
    print ""

    WA = Wavelet.Wavelet(A,'db4')
    print WA.wavedec('db4',1)