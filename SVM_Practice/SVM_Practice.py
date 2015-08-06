# -*- coding: utf-8 -*-
'''
Goal : SVM Classifier
Author : Yonghan Jung, ISyE, KAIST 
Date : 150806
Comment 
- 

'''

''' Library '''
import numpy as np
import matplotlib.pyplot as plt
from sklearn import svm, datasets
''' Function or Class '''


class Hans_SVM:
    def __init__(self):
        return None


if __name__ == "__main__":
    digits = datasets.load_digits()
    clf = svm.SVC(gamma = 0.001, C = 100)
    X = digits['data'][:-10]
    y = digits['target'][:-10]

    clf.fit(X,y)

    print('Predictoin', clf.predict(digits.data[-2]))
    print y.shape

    # plt.imshow(digits.images[-2], cmap = plt.cm.gray_r, interpolation = "nearest")
    # plt.show()



