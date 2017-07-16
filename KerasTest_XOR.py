# -*- coding: utf-8 -*-
"""
Created on Fri Jun 16 22:19:57 2017

@author: RaduGrig
@inspiration: NikCleju
"""

# import stuff
from keras.optimizers import Adam
from keras.models import Sequential
from keras.layers import Dense, Activation
import numpy as np
import matplotlib.pyplot as plot 

#Params
DataPoints = 100
Noise = 0.1

#create NN object
XoRNN = Sequential()
#
#Other models:
#???To add
#

#stack NN layers(build NN)
XoRNN.add(Dense(4, input_dim=2))
XoRNN.add(Activation('softmax'))
XoRNN.add(Dense(1))
XoRNN.add(Activation('linear'))
XoRNN.compile(optimizer=Adam(), loss="mse")



#
#Other activations:
#https://keras.io/activations/
#

#Create training data
Data00 = np.tile( np.array([0, 0]), (DataPoints, 1) ) + Noise * np.random.randn( DataPoints, 2 )
Data01 = np.tile( np.array([0, 1]), (DataPoints, 1) ) + Noise * np.random.randn( DataPoints, 2 )
Data10 = np.tile( np.array([1, 0]), (DataPoints, 1) ) + Noise * np.random.randn( DataPoints, 2 )
Data11 = np.tile( np.array([1, 1]), (DataPoints, 1) ) + Noise * np.random.randn( DataPoints, 2 )

Tests00 = np.tile( np.array([0, 0]), (DataPoints, 1) ) + Noise * np.random.randn( DataPoints, 2 )
Tests01 = np.tile( np.array([0, 0]), (DataPoints, 1) ) + Noise * np.random.randn( DataPoints, 2 )
Tests10 = np.tile( np.array([0, 0]), (DataPoints, 1) ) + Noise * np.random.randn( DataPoints, 2 )
Tests11 = np.tile( np.array([0, 0]), (DataPoints, 1) ) + Noise * np.random.randn( DataPoints, 2 )

Labels00 = np.zeros(( DataPoints, 1 ))
Labels01 = np.ones(( DataPoints, 1 ))
Labels10 = np.ones(( DataPoints, 1 ))
Labels11 = np.zeros(( DataPoints, 1 ))

TrainingSet = np.array( np.vstack((Data00, Data01, Data10, Data11)), dtype=np.float32)
TestSet = np.array( np.vstack((Tests00, Tests01, Tests10, Tests11)), dtype=np.float32)
Labels = np.vstack((Labels00, Labels01, Labels10, Labels11))

#Plot training set
plot.scatter(TrainingSet[:,0], TrainingSet[:,1], c=Labels)
plot.show()

#NikCleju's function for visualizing decission areas
def plot_separating_curve(model):
    points = np.array([(i, j) for i in np.linspace(0,1,100) for j in np.linspace(0,1,100)])
    #outputs = net(Variable(torch.FloatTensor(points)))
    outputs = model.predict(points)
    outlabels = outputs > 0.5
    plot.scatter(points[:,0], points[:,1], c=outlabels, alpha=0.5)
    plot.title('Decision areas')
    plot.show()

     
XoRNN.fit(TrainingSet, Labels, epochs=200, batch_size=50)
plot_separating_curve(XoRNN)