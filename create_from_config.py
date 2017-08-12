from dataSet import *
from deepLearner import *
from dataRepository import *
from keras.optimizers import SGD
import time
import json, yaml
from bs4 import BeautifulSoup
import sys

'''
Added code to extract parameters from various configuration file inputs: json, yaml, xml.

This will hopefully be the simplest possible way to configure a deep learner, and can also
act as a baseline to later DSL approaches...

Nina, 26 June 2017.
'''    

def isFloat(value):
	try:
		float(value)
		return True
	except ValueError:
		return False	

def isInt(value):
	try:
		int(value)
		return True
	except ValueError:
		return False
		
def toBool(val):
	print('val', val)
	if val=="True" or val=='True' or val==True:
		print('valfewfewew', val)
		return True
	else:
		return False			


start_time = time.time()
print("--- %s seconds ---" % (time.time() - start_time))


# Example 1: create deep learner from json file
def create_from_json(file):
	config = {}
	with open(file) as json_data_file:
		config = json.load(json_data_file)
	for c in config['setup']:
		config['setup'][c] = toBool(config['setup'][c])
	return config	
    
    
# Example 2: create deep learner from yaml file    
def create_from_yaml(file):
	config = {}
	with open(file, 'r') as ymlfile:
		config = yaml.load(ymlfile)
	for c in config['setup']:
		config['setup'][c] = toBool(config['setup'][c])		
	return config	


# Example 3: create deep learner from xml file  
# (slightly more complicated)
def create_from_xml(file):

	config = {}
	with open(file) as f:
		content = f.read()

	y = BeautifulSoup(content, "lxml")
	dl_parameters = {}
	dataset = {}
	setup = {}	
	hyperparameter_setup = {}	
	
	for element in y.dl_parameters:
		if not element=='\n':
			if isInt(element.contents[0]):
				dl_parameters[element.name] = int(element.contents[0])
			elif isFloat(element.contents[0]):
				dl_parameters[element.name] = float(element.contents[0])			
			else:
				dl_parameters[element.name] = element.contents[0]
				
	for element in y.dataset:
		if not element=='\n':
			dataset[element.name] = element.contents[0]

	for element in y.x_setup:
		if not element=='\n':
			setup[element.name] = element.contents[0]	
			
	for element in y.hyperparameter_setup:
		if not element=='\n':
			hyperparameter_setup[element.name] = element.contents[0]					
			
	config['dl_parameters']	= dl_parameters
	config['dataset']	= dataset

	config['setup']	= setup	
	for c in config['setup']:
		config['setup'][c] = toBool(config['setup'][c])						
			
	config['hyperparameter_setup']	= hyperparameter_setup

	return config

################################################################

# Here the input file please...  

file = 'config.json'

################################################################


# Parameters to be read in from configuration file...
config = {}

if file.endswith('.json'):
	print('Reading configuration from file', file, '...')
	config = create_from_json(file)
elif file.endswith('.yml'):
	print('Reading configuration from file', file, '...')	
	config = create_from_yaml(file)		
elif file.endswith('.xml'):
	print('Reading configuration from file', file, '...')	
	config = create_from_xml(file)		
else:
	print('Error - input file needs to be one of .json, .yml or .xml')		
				

# From hereon using information provided to arrange data and create a learning agent.
data_repo = DataRepository()
data_repo.load(config['dataset']['path']) 
data = DataSet(data_repo.X_set, data_repo.Y_set, config['dl_parameters']['prediction'])
data.representData().shuffleAndSplit()	
	
deep = defineDL(config['dl_parameters'])

	
if config['setup']['hyperparameters']:

	print('Optimising hyperparameters...')
		
	def create_model4hyperOpt(learning_rate=deep.learning_rate, momentum=deep.momentum, init_mode=deep.init_mode, activation1=deep.activation1, activation2=deep.activation2, dropout_rate=deep.dropout_rate, weight_constraint=deep.weight_constraint, hidden_size=deep.hidden_size, loss=deep.loss, optimiser=deep.optimiser, epochs=deep.epochs, batch_size=deep.batch_size, layers=deep.layers, model_string=deep.model_string):			
		deep.designModel(data).compileModel()
		return deep.model		
										
	best = deep.hyperOptimise(create_model4hyperOpt, data.X_set, data.Y_set, ['optimiser', 'layers', 'dropout_rate', 'weight_constraint', 'hidden_size', 'learning_rate', 'momentum', 'init_mode', 'activation_1', 'activation_2', 'loss', 'epochs', 'batch_size'], verbose=config['hyperparameter_setup']['verbose'], search=config['hyperparameter_setup']['search'], multithreaded=config['setup']['multithreaded'], logging=config['hyperparameter_setup']['logging'])

	deep.updateParameters(best[1])
	
	if config['setup']['train']:
		print('Training with updated hyperparameters...')
		deep.trainModel(data, output_file=config['dataset']['weights_out'], verbose=config['setup']['verbose'])	
	
else:
	deep.designModel(data).compileModel()

	if config['setup']['train']:				
		print('Training with specified parameters...')
		deep.trainModel(data, output_file=config['dataset']['weights_out'], verbose=config['setup']['verbose'])

print("--- %s seconds ---" % (time.time() - start_time))

