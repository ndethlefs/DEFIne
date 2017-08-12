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
Some code that tests the new dataRepository class using data from here: http://archive.ics.uci.edu/ml/

Datasets include some classification tasks and regression tasks. 

Should all run on MLP, recurrent models (RNN, LSTM and GRU) in either a multithreaded or single-threaded setting. 

Nina, 13 June 2017. 

'''    

data_repo = DataRepository()
# Binary classification tasks (work for MLPs and recurrent models on one or multiple threads).
#data_repo.load('cleveland') 
#data_repo.load('breast-cancer') 

# Classification task with multiple categorical outputs (work for MLPs and recurrent models on one or multiple threads).
#data_repo.load('iris') 
#data_repo.load('abalone') 
#data_repo.load('wine') 
data_repo.load('winequality-red') 
#data_repo.load('winequality-white') 
#data_repo.load('uci_har')
#data_repo.load('poker') 

# Binary classification tasks with a string output rather than an int. Work on MLPs and recurrent models, one or multiple threads.
#data_repo.load('bank') 
#data_repo.load('car') 
#data_repo.load('adult') 

# Regression task. Works with MLPs and recurrent models.
#data_repo.load('forestfires') 
#data_repo.load('abalone') # may need an additional distinction in repository (classification / regression)


print(data_repo.X_set)
print(data_repo.X_set.shape)
print(data_repo.Y_set)
print(data_repo.Y_set.shape)
#sys.exit(0)

X_set = np.asarray(data_repo.X_set)
Y_set = np.asarray(data_repo.Y_set)

print(X_set.shape)
print(Y_set.shape)
print('ss', set(Y_set))
#sys.exit(0)

print('-'*50)
print('Designing model: ***', 'heart_disease', '***')
print('(DataSet)')

start_time = time.time()
# analyse data and represent as needed, count some items we need to know.
data = DataSet(X_set, Y_set, 'classification')
data.representData().shuffleAndSplit()


print('-'*50)		

# now choose and compile a deep learning model, train it and save the outputs.
print('(DeepLearner)')

parameters = {
'model_name' : 'heart_disease_cpu_gru',
'modelString' : 'MLP',
'prediction' : data.prediction
}		


# Use this to train with given parameters.
		
#defineDL(parameters).designModel(data).compileModel().trainModel(data, output_file='model-weights.h5', verbose=False)


# Use this to do hyper-parameter optimisation. 
# Method create_model4hyperOpt takes a list of parameters to optimise.

deep = defineDL(parameters)		

def create_model4hyperOpt(learning_rate=deep.learning_rate, momentum=deep.momentum, init_mode=deep.init_mode, activation1=deep.activation1, activation2=deep.activation2, dropout_rate=deep.dropout_rate, weight_constraint=deep.weight_constraint, hidden_size=deep.hidden_size, loss=deep.loss, optimiser=deep.optimiser, epochs=deep.epochs, batch_size=deep.batch_size, layers=deep.layers, modelString=deep.modelString):			
	deep.designModel(data).compileModel()
	return deep.model

best = deep.hyperOptimise(create_model4hyperOpt, data.X_set, data.Y_set, ['optimiser', 'layers', 'dropout_rate', 'weight_constraint', 'hidden_size', 'learning_rate', 'momentum', 'init_mode', 'activation_1', 'activation_2', 'loss', 'epochs', 'batch_size'], verbose=1, search='random', multithreaded=True)
#deep.updateParameters(best[1])
#deep.trainModel(data, output_file='heart_disease-weights.h5', verbose=False)


print("--- %s seconds ---" % (time.time() - start_time))


