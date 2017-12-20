from dataSet import *
from deepLearner import *
from dataRepository import *
import os
import numpy as np
from keras.optimizers import SGD
import time
from keras.datasets import mnist
import sys

'''
Some code that creates an MLP to train on eeg eyes classification dataset.

Nina, 20 Dec 2017. 

'''    

data_repo = DataRepository()
data_repo.load('eeg_eyes.txt') 

X_set = np.asarray(data_repo.X_set)
Y_set = np.asarray(data_repo.Y_set)

print('-'*50)
print('Designing model: ***', 'eeg_eyes', '***')
print('(DataSet)')

start_time = time.time()
# analyse data and represent as needed, count some items we need to know.
data = DataSet(X_set, Y_set, 'classification')
data.representData().shuffleAndSplit()


print('-'*50)		

# now choose and compile a deep learning model, train it and save the outputs.
print('(DeepLearner)')

parameters = {
'model_name' : 'eeg_eyes_mlp',
'modelString' : 'MLP',
'prediction' : data.prediction
}		


# Use this to train with given parameters.
		
defineDL(parameters).designModel(data).compileModel().trainModel(data, output_file='model-weights.h5', verbose=False)

print("--- %s seconds ---" % (time.time() - start_time))


