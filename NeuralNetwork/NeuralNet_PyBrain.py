# -*- coding: utf-8 -*-
'''
Goal : Neural Net using Pybrain
Author : Yonghan Jung, ISyE, KAIST 
Date : 150806
Comment 
- 

'''

''' Library '''
# Dataset
from sklearn import datasets
import numpy as np

# Pybrain module
from pybrain.datasets            import ClassificationDataSet
from pybrain.utilities           import percentError
from pybrain.tools.shortcuts     import buildNetwork
from pybrain.supervised.trainers import BackpropTrainer
from pybrain.structure.modules   import SoftmaxLayer
from pybrain.tools.xml.networkwriter import NetworkWriter
from pybrain.tools.xml.networkreader import NetworkReader




''' Function or Class '''




if __name__ == "__main__":
    X = datasets.load_iris()['data']
    y = datasets.load_iris()['target']

    Dim = X.shape[1]
    NNData = ClassificationDataSet(Dim)

    for Idx in range(len(X)):
        NNData.addSample(np.ravel(X[Idx]), y[Idx])

    TrainData, TestData = NNData.splitWithProportion(0.25)
    TrainData._convertToOneOfMany()
    TestData._convertToOneOfMany()
    print TrainData.indim
    print TrainData.outdim

    HiddenNum = int(len(TrainData)/ float(2 * (TrainData.indim + TrainData.outdim)))
    print HiddenNum

    NNNetwork = buildNetwork(TrainData.indim, HiddenNum, TrainData.outdim, outclass = SoftmaxLayer)
    trainer = BackpropTrainer( NNNetwork, dataset=TrainData, momentum=0.1, learningrate=0.01 , verbose=True, weightdecay=0.01)
    trainer.trainEpochs(100)
    MyAnswer =  trainer.testOnClassData(dataset=TestData)

    for a,b in zip(MyAnswer, TestData['target']):
        print a,b


    # olivetti = datasets.fetch_olivetti_faces()
    # X, y = olivetti.data, olivetti.target
    #
    # Dim = X.shape[1]
    #
    #
    # ds = ClassificationDataSet(Dim, 1 , nb_classes=40)
    # for k in xrange(len(X)):
    #     ds.addSample(np.ravel(X[k]), y[k])
    # tstdata, trndata = ds.splitWithProportion( 0.25 )
    # trndata._convertToOneOfMany( )
    # tstdata._convertToOneOfMany( )
    #
    # print trndata['input'], trndata['target'], tstdata.indim, tstdata.outdim
    # fnn = buildNetwork( trndata.indim, 64 , trndata.outdim, outclass=SoftmaxLayer )
    # trainer = BackpropTrainer( fnn, dataset=trndata, momentum=0.1, learningrate=0.01 , verbose=True, weightdecay=0.01)
    # trainer.trainEpochs (1)
    # print 'Percent Error on Test dataset: ' , percentError( trainer.testOnClassData (
    #        dataset=tstdata ), tstdata['class'] )






