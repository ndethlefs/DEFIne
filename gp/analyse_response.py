from dataSet import *
from dataRepository import *
import numpy as np
import sys
import json

'''
Lazy way to produce knowledge.json files for hyperparameter transfer.
'''


def all_numeric(labels):
	print("Test if outputs are all numeric.")
	types = [isinstance(n, numbers.Number) for n in labels]
	if not False in types:
		return True
	else:
		return False	


features = {
1 : 'search',
2 : 'init_mode',
3 : 'loss',
4 : 'modelString',
5 : 'epochs',
6 : 'learning_rate',
7 : 'batch_size',
8 : 'dropout_rate',
9 : 'hidden_size',
10 : 'layers',
11 : 'optimiser',
12 : 'momentum',
13 : 'activation1',
14 : 'activation2',
15 : 'weight_constraint',
16 : 'fit_time',
17 : 'cpu_cores',
18 : 'search_algorithm',
19 : 'search_space',
20 : 'hardware',
21 : 'learning_tasks',
22 : 'size_of_datasets',
23 : 'input_features',
24 : 'output_features',
25 : 'data_type',
26 : 'vocab_size',
27 : 'dimensionality',
28 : 'score',
}	


print(features)

dict = {}

infile_gp = open("results_regression.txt", "r")

counter = 1

for line in infile_gp:
	if line.replace("\n", "").endswith(".txt"):
		counter = 1
		name = line.replace("\n", "")
		dict[name] = {}
	if line.startswith("beeest"):
		dict[name][features[counter]] = line.replace("\n", "").replace("beeest1**2 * RBF", "")
		counter = counter + 1

print(dict)
print(len(dict))

infile_gp.close()

def normalized(a, axis=-1, order=2):
    l2 = np.atleast_1d(np.linalg.norm(a, order, axis))
    l2[l2==0] = 1
    return a / np.expand_dims(l2, axis)



for key in dict:
	print("Analysing dataset:", key)
	data_vector = []
	infile_stats = key
	data_repo = DataRepository()
	data_repo.load(infile_stats) 		
	X_set = np.asarray(data_repo.X_set)
	Y_set = np.asarray(data_repo.Y_set)
	data = DataSet(X_set, Y_set, 'regression')
	data.representData()
		
	prediction = 1
	input_features = data.max_len	
	output_features = data.outputs
#	dimensionality = data.dimensionality
	data_type = data.data_type
	instances = len(data.X_set)
	vocab_size = len(data.vocab)
	mean = data.mean_X
	median = data.median_X
	std = data.stds_x
	iqr = data.iqr_x
	skew = data.skew_x
	kurtosis = data.kurtosis_x
	label_distribution = data.label_distribution
	normal_distribution = data.normal_distribution
	print('normal', normal_distribution)
	if normal_distribution == True:
		normal_distribution = 1
	else:	
		normal_distribution = 0
	
	if all_numeric(set(data_repo.Y_set)):
		if(len(set(data_repo.Y_set))) < 11:
			prediction = 1
		else:
			prediction = 0 

	if data_type == "numeric":
		data_type = 0
	else:
		data_type = 1			
			
	print("prediction", prediction)			
	data_vector.append(prediction)
	print("input_features", input_features)
	data_vector.append(input_features)	
	print("output_features", output_features)	
	data_vector.append(output_features)		
#	print("dimensionality", dimensionality)
	print("data_type", data_type)
	data_vector.append(data_type)	
	print("instances", instances)
	data_vector.append(instances)		
	print("vocab_size", vocab_size)
	data_vector.append(vocab_size)
	print("mean", mean)
	data_vector.append(mean)	
	print("median", median)
	data_vector.append(median)	
	print("std", std)
	data_vector.append(std)	
	print("iqr", iqr)
	data_vector.append(iqr)	
	print("skew", skew)
	data_vector.append(skew)	
	print("kurtosis", kurtosis)
	data_vector.append(kurtosis)	
	print("label_distribution", label_distribution)	
	print("normal_distribution", normal_distribution)
	data_vector.append(normal_distribution)	
			
	normalised = normalized(data_vector)
	dict[key]['normalised'] = str(normalised)
	print(key, dict[key])
			
			
with open('knowledge_regression.json', 'w') as k:
    json.dump(dict 	, k)








