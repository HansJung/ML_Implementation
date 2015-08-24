# -*- coding: utf-8 -*-
'''
Goal : Bootstrap 을 통한 SHT, CUSUM 분포추정
Author : Yonghan Jung, ISyE, KAIST 
Date : 150824
Comment 
- 

'''

''' Library '''
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import gaussian_kde

''' Function or Class '''


class Example:
    def __init__(self):
        return None

# def Generate_Data():


if __name__ == "__main__":
    P = 1.0
    M = 300.0
    T = 10
    Bootstrap_Iter = 1000
    SamplingIter = 100

    QuantileDict = dict()
    for x in range(40):
        QuantileValue = 90 + (x/4.0)
        QuantileDict[QuantileValue] = list()

    for SampleIdx in range(SamplingIter):
        SampleBox = list()
        for idx in range(Bootstrap_Iter):
            # T sample generation
            FSamples = np.random.f(P, M, T)
            # print FSamples
            TSamples = ((P * ((M-1) ** 2)) / (M * (M-P))) * FSamples
            NewSample = np.sum(TSamples)
            SampleBox.append(NewSample)
        for DictIdx, DictKey in enumerate(sorted(QuantileDict)):
            QuantileDict[DictKey].append(np.percentile(SampleBox, DictKey))

    for idx, key in enumerate(sorted(QuantileDict)):
        print np.mean(QuantileDict[key])
    # print np.percentile(SampleBox, 99.99)
    # print np.max(SampleBox)
    # Density = gaussian_kde(SampleBox)
    # Domain = np.linspace(np.min(SampleBox)-10, np.max(SampleBox) + 10, 1000)
    # plt.plot(Domain, Density(Domain))
    # plt.show()