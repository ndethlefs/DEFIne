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
Code to train a deep learner with hyper-parameter optimisation using only those hyper-parameters 
that were good predictors of performance in the most similar dataset in previous experiments.

It will gather basic statistics about the target dataset: inputs, outputs, instances, distribution, skew etc.
Then it will look into knowledge.json and find (based on the "normalised" vector) whatever the most similar dataset is.
It then inherits those hyper-parameters for optimisation that had a length scale of no more than 3.

Nina, 22 Nov 2017.
'''    

data_repo = DataRepository()
data_repo.load('imdb_out.txt') 

X_set = np.asarray(data_repo.X_set)
Y_set = np.asarray(data_repo.Y_set)


start_time = time.time()
data = DataSet(X_set, Y_set, 'classification') 
data.representData().shuffleAndSplit()

print('-'*50)		

# now choose and compile a deep learning model, train it and save the outputs.
print('(DeepLearner)')

parameters = {
'model_name' : 'heart',
'model_string' : 'MLP',
'prediction' : data.prediction
}		

test = data.getNormalisedVector()
print('test',test)
sys.exit(0)

json_data=open("knowledge_regression.json").read()
kdata = json.loads(json_data)

#for d in kdata:
#	i = getImportantParameters(kdata[d])
#	print(d, "=", i)
#	v = getDataVector(kdata[d])
#	print(v)	
#	print('euclidean', euclideanDistance(test, v))
#	print('cosine', cosineDistance(test, v))


deep = defineDL(parameters).designModel(data).compileModel()

dataset, similar = deep.getMostSimilarDataset(test, kdata)
#dataset, similar = getMostSimilarDataset(test, kdata)
#print(dataset, similar)
i = deep.getImportantParameters(kdata[dataset])



# Use this to train with given parameters.
		
#deep = defineDL(parameters).designModel(data).compileModel().trainModel(data, output_file='model-weights.h5', verbose=False)


#deep.trainModel(data, output_file='out-weights.h5', verbose=False)


print("--- %s seconds ---" % (time.time() - start_time))


