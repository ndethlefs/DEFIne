from keras.models import Sequential
from keras.layers import Activation, TimeDistributed, Dense, RepeatVector, recurrent, Embedding, Flatten, Reshape, Dropout
import datetime, time
import numpy as np
from keras.constraints import maxnorm
from keras.optimizers import SGD, Adam
from keras.wrappers.scikit_learn import KerasClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import RandomizedSearchCV
import multiprocessing
import json
import sys

class DeepLearner:

	def __init__(self, model_string, layers=2, batch_size=32, epochs=50,  dropout_rate=0.0, weight_constraint=5, hidden_size=128, embedding_size=1000, learning_rate = 0.01, momentum= 0.9, optimiser='adam', init_mode='uniform', activation1='relu', activation2='sigmoid', loss="mean_squared_error", eval_metrics=['accuracy'], model_name='model_1', prediction='regression'):


		self.model_string = model_string
		self.model_name = model_name
		self.model = ''
		self.layers = layers		
		self.batch_size = batch_size
		self.epochs = epochs
		self.hidden_size = hidden_size
		self.embedding_size = embedding_size
		self.learning_rate = learning_rate
		self.momentum = momentum		
		self.loss = loss		
		self.eval_metrics = eval_metrics
		self.dropout_rate = dropout_rate
		self.weight_constraint = weight_constraint		
		self.optimiser = Adam(learning_rate, momentum)		
		self.init_mode = init_mode
		self.activation1 = activation1
		self.activation2 = activation2		
		self.prediction = prediction
		

	def designModel(self, dataset):		
		# Based on data representation, choose the best model to design. 
		
		if dataset.dimensionality=='3D':
			self.designSeq2Seq(dataset.max_len, dataset.vocab)			
		# two paths for 2D models, recurrent or not.	
		elif self.model_string=='MLP':
			self.designMLP(dataset.max_len, dataset.outputs)	
		else:
			self.designRNN(dataset.max_len, dataset.outputs)
			
		return self			
		
		
	def designSeq2Seq(self, max_len, vocab):
		# Puts together a model that learns to map an input sequence to an output sequence. Requires 3D input matrices.
		print('Designing a(n) sequence-to-sequence', self.model_string, 'with', self.layers, 'layers,', self.hidden_size, 'hidden nodes, and', self.activation1, 'activation.')
		
		RNN = recurrent.LSTM
		if self.model_string=='LSTM':
			RNN = recurrent.LSTM	
		elif self.model_string=='RNN':
			RNN = recurrent.SimpleRNN
		elif self.model_string=='GRU':
			RNN = recurrent.GRU
		
		self.model = Sequential()
		self.model.add(RNN(self.hidden_size, input_shape=(max_len, len(vocab))))
		self.model.add(RepeatVector(max_len))
		for _ in range(self.layers):
		    self.model.add(RNN(self.hidden_size, return_sequences=True))
		self.model.add(TimeDistributed(Dense(len(vocab))))
		self.model.add(Activation(self.activation1))

#		self.model.summary()		            	  
		
		return self.model


	def designRNN(self, max_len, outputs):
		# Puts together a model that learns to map an input sequence to a single output value. Requires 2D input matrices.
		print('Designing a(n) ', self.model_string, 'with', self.layers, 'layers,', self.hidden_size, 'hidden nodes, and', self.activation1, ' activation.')
		
		RNN = recurrent.LSTM
		if self.model_string=='LSTM':
			RNN = recurrent.LSTM	
		elif self.model_string=='RNN':
			RNN = recurrent.SimpleRNN
		elif self.model_string=='GRU':
			RNN = recurrent.GRU	
		
		
		self.model = Sequential()
		self.model.add(Embedding(self.embedding_size, self.hidden_size, input_length=max_len))
		self.model.add(Dropout(self.dropout_rate))		
		self.model.add(RNN(self.hidden_size, recurrent_dropout=0.2, dropout=0.2))  
		self.model.add(RepeatVector(max_len))
		for _ in range(self.layers):
		    self.model.add(RNN(self.hidden_size, return_sequences=True))		
		self.model.add(Flatten())
		if self.prediction=='regression':
			self.model.add(Dense(1))		
		else:	
			self.model.add(Dense(outputs))
		self.model.add(Activation(self.activation1))

#		self.model.summary()		            	  
		
		return self.model

		
	def designMLP(self, max_len, outputs):
		# Puts a model together that learns to map an input sequence to an output sequence. Requires 3D input matrices.
		print('Designing a(n) ', self.model_string, 'with', self.layers, 'layers,', self.hidden_size, 'hidden nodes, and', self.activation1, ' activation.')

		self.model = Sequential()
		self.model.add(Dense(self.hidden_size, input_dim=max_len, kernel_initializer=self.init_mode, activation=self.activation1 ))
		self.model.add(Dropout(self.dropout_rate))		
		if self.layers == 2:
			self.model.add(Dense(self.hidden_size, input_dim=max_len, kernel_initializer=self.init_mode, activation=self.activation1 ))		
			self.model.add(Dropout(self.dropout_rate))				
		if self.layers == 3:
			self.model.add(Dense(self.hidden_size, input_dim=max_len, kernel_initializer=self.init_mode, activation=self.activation1 ))		
			self.model.add(Dropout(self.dropout_rate))							
		if self.layers > 3:
			print('Not doing MLPs with more than 3 layers now - it is no use anyway...')			
		if self.prediction=='regression':	
			self.model.add(Dense(1, kernel_initializer=self.init_mode, activation=self.activation2))
		else:	
			self.model.add(Dense(outputs, kernel_initializer=self.init_mode, activation=self.activation2))
	
		self.model.compile(loss=self.loss, optimizer=self.optimiser, metrics=self.eval_metrics)		

		print(self.model.summary())
	
#		self.model.summary()		            	  
		
		return self.model		
		
		
	def designMLP3D(self, max_len):
		# Puts a model together that learns to map an input sequence to an output sequence. Requires 3D input matrices.
		print('Designing a(n) ', self.model_string, 'with', self.layers, 'layers,', self.hidden_size, 'hidden nodes, and', self.activation1, 'activation.')

		self.model = Sequential()
		self.model.add(Dense(self.hidden_size, input_shape=(max_len,)))
		self.model.add(Activation(self.activation1))
		self.model.add(Dropout(0.2))
		self.model.add(Dense(512))
		self.model.add(Activation('relu'))
		self.model.add(Dropout(0.2))
		self.model.add(Dense(10))
		self.model.add(Activation(self.activation2))		

#		self.model.summary()		            	  
		
		return self.model			
		
	def compileModel(self):
		# Compiles the model ready for training or testing with the chosen optimisation and training parameters. 
		print('Model is ready to train (or test), using', self.loss, ',', self.optimiser, 'optimisation, and evaluating', self.eval_metrics, '.')
		if self.prediction =='regression':
			self.loss = 'mean_absolute_error'
		self.model.compile(loss=self.loss,
        	      optimizer=self.optimiser,
            	  metrics=self.eval_metrics)	
		return self            	  
            	  
            	  
	def loadWeights(self, model, file):
		print("Loading pre-trained weights from ", file)
		
		self.model.load_weights(file)
		return self.model.get_weights()
            	  


	def trainModel(self, data, output_file='out.txt', verbose=True):
	
		if verbose:
			self.trainVerbose(data.X_train, data.Y_train, data.X_val, data.Y_val, data.indices_char, data.dimensionality, output_file)
		else:
			self.train(data.X_train, data.Y_train, data.X_val, data.Y_val, output_file)
            	  
            	  
	def train(self, X_train, Y_train, X_val, Y_val, output_file='out.txt'):
		# Actually trains for a given number of epochs.
		
		#print X_train
		print('Started training at:', (time.strftime("%d/%m/%Y")), (time.strftime("%H:%M:%S")))	
	
		for iteration in range(1, self.epochs):
		    print()
		    print('-' * 50)
		    print('Iteration', iteration)
		    self.model.fit(X_train, Y_train, batch_size=self.batch_size, nb_epoch=1, validation_data=(X_val, Y_val))
		    score, acc = self.model.evaluate(X_val, Y_val, batch_size=self.batch_size)                           
		    json_string = self.model.to_json()
		    self.model.save_weights(output_file, overwrite=True)		    
                          
		    print('Test score:', score)
		    print('Test accuracy:', acc)     	           	  			
            	  
		print('Finished at:', (time.strftime("%d/%m/%Y")), (time.strftime("%H:%M:%S")))            	  


	def decode3D(self, X, indices_char, calc_argmax=True):
		# this decodes 3D items. 
		if calc_argmax:
			X = X.argmax(axis=-1)
		return ' '.join(indices_char[x] for x in X)	
		
	
	def decode2D(self, X_item, indices_char):
		# this decodes 2D X items. 
		new_X = []
		for x in X_item:
			new = indices_char[int(x)]
			new_X.append(new)
		new_X = ' '.join(new_X).replace('mask_zeros', '')
		return ' '.join(new_X.split())			            	  


	def trainVerbose(self, X_train, Y_train, X_val, Y_val, indices_char, dimensionality, output_file='out.txt'):
		if dimensionality=='3D':
			self.trainVerbose3D(X_train, Y_train, X_val, Y_val, indices_char, output_file)
		else:
			self.trainVerbose2D(X_train, Y_train, X_val, Y_val, indices_char, output_file)						


		            	  
	def trainVerbose2D(self, X_train, Y_train, X_val, Y_val, indices_char, output_file='out.txt'):
		# Actually trains for a given number of epochs and prints some example tests after each epoch.
		
		#print X_train
		print('Started training at:', (time.strftime("%d/%m/%Y")), (time.strftime("%H:%M:%S")))	
		numeric = False
		if type(X_train) is np.ndarray:
			numeric = True
	
		for iteration in range(1, self.epochs):
		    print()
		    print('-' * 50)
		    print('Iteration', iteration)
		    self.model.fit(X_train, Y_train, batch_size=self.batch_size, nb_epoch=1, validation_data=(X_val, Y_val))
		    score, acc = self.model.evaluate(X_val, Y_val, batch_size=self.batch_size)
                            
                            
		    # Select 10 samples from the validation set at random to visualise and inspect. 
		    for i in range(10):
		        ind = np.random.randint(0, len(X_val))
		        rowX, rowy = X_val[np.array([ind])], Y_val[np.array([ind])]
		        print(rowX, rowy)
		        preds = self.model.predict_classes(rowX, verbose=0)
		        if numeric:
		        	q = rowX[0]
		        else:	
			        q = self.decode2D(rowX[0], indices_char)
		        print(rowy[0], preds[0])    
		        correct = int(rowy[0])
		        guess = int(preds[0])
		        print()
		        print('Input vector: ', q)
		        print('Correct label: ', correct)
		        print('Predicted label: ' + str(guess) + ' (good)' if correct == guess else 'Predicted label: ' + str(guess) + ' (bad)' )
		        print('---')
		        json_string = self.model.to_json()
		        self.model.save_weights(output_file, overwrite=True)                             
                          
		    print('Test score:', score)
		    print('Test accuracy:', acc)     	           	  			
            	  
		print('Finished at:', (time.strftime("%d/%m/%Y")), (time.strftime("%H:%M:%S")))

            	  
	def trainVerbose3D(self, X_train, Y_train, X_val, Y_val, indices_char, output_file='out.txt'):
		# Actually trains for a given number of epochs and prints some example tests after each epoch.
		
		#print X_train
		print('Started training at:', (time.strftime("%d/%m/%Y")), (time.strftime("%H:%M:%S")))	
	
		for iteration in range(1, self.epochs):
		    print()
		    print('-' * 50)
		    print('Iteration', iteration)
		    self.model.fit(X_train, Y_train, batch_size=self.batch_size, nb_epoch=1, validation_data=(X_val, Y_val))
		    score, acc = self.model.evaluate(X_val, Y_val, batch_size=self.batch_size)
                            
		    # Select 10 samples from the validation set at random to visualise and inspect. 
		    for i in range(10):
		        ind = np.random.randint(0, len(X_val))
		        rowX, rowy = X_val[np.array([ind])], Y_val[np.array([ind])]
		        preds = self.model.predict_classes(rowX, verbose=0)
		        q = self.decode3D(rowX[0], indices_char)
		        correct = self.decode3D(rowy[0], indices_char)
		        guess = self.decode3D(preds[0], indices_char, calc_argmax=False)
		        print()
		        print('Input vector: ', q)
		        print('Correct label: ', correct)
		        print('Predicted label: ' + str(guess) + ' (good)' if correct == guess else 'Predicted label: ' + str(guess) + ' (bad)' )
		        print('---')
		        json_string = self.model.to_json()
		        self.model.save_weights(output_file, overwrite=True)                             
                          
		    print('Test score:', score)
		    print('Test accuracy:', acc)     	           	  			
            	  
		print('Finished at:', (time.strftime("%d/%m/%Y")), (time.strftime("%H:%M:%S")))
		

	def hyperOptimise(self, model, X_set, Y_set, paras4Op, verbose=False, search='grid', multithreaded=True, logging=True, cv=5, n_iter=50):

		start_time = time.time()	
		if multithreaded==True:
			n_jobs = -1
		else:
			n_jobs = 1	
		# picking up some values in case config file got type wrong...	
		if verbose=='True' or verbose==True:
			verbose = 1
		else:
			verbose = 0	
		if logging=='True' or logging==True:
			logging=True
		else:
			logging = False			
		kerasmodel = KerasClassifier(build_fn=model, verbose=verbose)
		if 'init_mode' in paras4Op:
			init_mode = ['uniform', 'lecun_uniform', 'normal', 'zero', 'glorot_normal', 'glorot_uniform', 'he_normal', 'he_uniform']
		else:
			init_mode = [self.init_mode]
		if 'weight_constraint' in paras4Op:	
			weight_constraint = [1, 2, 3, 4, 5]		
		else:
			weight_constraint = [self.weight_constraint]	
		if 'dropout_rate' in paras4Op:				
			dropout_rate = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
		else:
			dropout_rate = [self.dropout_rate]	
		if 'learning_rate' in paras4Op:
			learning_rate = [0.001, 0.01, 0.1, 0.2, 0.3]
		else:
			learning_rate = [self.learning_rate]	
		if 'momentum' in paras4Op:
			momentum = [0.0, 0.2, 0.4, 0.6, 0.8, 0.9]
		else:
			momentum = [self.momentum]	
		if 'hidden_size' in paras4Op:
			hidden_size = [1, 5, 10, 15, 20, 25, 30, 50, 100, 150, 200]
		else:
			hidden_size = [self.hidden_size]
		if 'batch_size' in paras4Op:
			batch_size= [10, 20, 40, 60, 80, 100]
		else:
			batch_size = [self.batch_size]
		if 'epochs' in paras4Op:
			epochs = [10, 20, 50, 100, 200]
		else:
			epochs = [self.epochs]			
		if 'activation1' in paras4Op:
			activation1 = ['softmax', 'softplus', 'softsign', 'relu', 'tanh', 'sigmoid', 'hard_sigmoid', 'linear']
		else:
			activation1 = [self.activation1]	
		if 'activation2' in paras4Op:
			activation2 = ['softmax', 'softplus', 'softsign', 'relu', 'tanh', 'sigmoid', 'hard_sigmoid', 'linear']
		else:
			activation2 = [self.activation2]				
		if 'optimiser' in paras4Op:
#			optimiser = ['Adam', 'SGD']
			optimiser = ['SGD', 'RMSprop', 'Adagrad', 'Adadelta', 'Adam', 'Adamax', 'Nadam']			
		else:
			optimiser = [Adam(learning_rate, momentum)]
		if 'loss' in paras4Op:
			loss = ['mean_squared_error', 'mean_absolute_error', 'mean_absolute_percentage_error', 'mean_squared_logarithmic_error', 'squared_hinge, hinge', 'logcosh', 'categorical_crossentropy', 'sparse_categorical_crossentropy', 'binary_crossentropy', 'kullback_leibler_divergence', 'poisson', 'cosine_proximity']		
		else:
			loss = [self.loss]	
		if 'layers' in paras4Op:
			layers = [1, 2, 3]
		else:
			layers = [self.layers]				
		if 'model_string' in paras4Op:
			model_string = ['RNN', 'LSTM', 'GRU']		
		else:
			model_string = [self.model_string]				
						

		param_grid = dict(optimiser=optimiser, init_mode=init_mode, weight_constraint=weight_constraint, dropout_rate=dropout_rate, learning_rate=learning_rate, momentum=momentum, hidden_size=hidden_size, batch_size=batch_size, epochs=epochs, activation1=activation1, activation2=activation2, loss=loss, layers=layers, model_string=model_string)
		# add optimiser=optimiser to list, doesn't work currently
				
#		grid = GridSearchCV(estimator=kerasmodel, param_grid=param_grid, n_jobs=-1)
		if search=='random':
			search_algorithm = RandomizedSearchCV(estimator=kerasmodel, param_distributions=param_grid, n_jobs=n_jobs, cv=cv, n_iter=n_iter)		
		else:	
			search_algorithm = GridSearchCV(estimator=kerasmodel, param_grid=param_grid, n_jobs=n_jobs, cv=cv)				
	
		
		search_space = len(init_mode) * len(weight_constraint) * len(dropout_rate) * len(optimiser) * len(batch_size) * len(epochs) * len(learning_rate) * len(momentum) * len(activation1) * len(activation2) * len(loss) * len(hidden_size) * len(layers) * len(model_string) 
		print('Using all possible parameters, searching a space of', search_space, ' options...')
			
		print('X', X_set.shape)	
		print('Y', Y_set.shape)	
		search_result = search_algorithm.fit(X_set, Y_set)

		print("Best: %f using %s" % (search_result.best_score_, search_result.best_params_))
		print("Optimised parameters", param_grid, '.')
		means = search_result.cv_results_['mean_test_score']
		fit_times = search_result.cv_results_['mean_fit_time']		
		stds = search_result.cv_results_['std_test_score']
		params = search_result.cv_results_['params']
		for mean, stdev, param in zip(means, stds, params):
			print("%f (%f) with: %r" % (mean, stdev, param))
			
		if logging==True:
			self.logHyperParameters(self.model_name, search, search_space, start_time, (search_result.best_score_, search_result.best_params_), zip(means, params, fit_times))	

		return (search_result.best_score_, search_result.best_params_)
		
		
	def logHyperParameters(self, model_name, search_algorithm, search_space, start_time, best_search, all_searches):
	
		
		outfile = open("./logs/"+model_name+".json", 'w')
		
		# make sure models come out correctly... 
		
		general_parameters = {}
		general_parameters['cpu_cores'] = multiprocessing.cpu_count()
		general_parameters['search_algorithm'] = search_algorithm
		general_parameters['search_space'] = search_space
		general_parameters['best_result'] = best_search[0]
		
		best_parameters = best_search[1]
		for key in best_parameters:
			general_parameters['best_'+key] = best_parameters[key]
				
		x = 0				
		for mean, param, fit_time in all_searches:
			x = x+1
			m_name = 'search_'+str(x)
			current_score = mean
			current_parameters = param
			dict = {}
			dict['score'] = mean
			dict['fit_time'] = fit_time
			dict['param'] = param
			general_parameters[m_name] = dict
			
		
		run_time = (time.time() - start_time)
		general_parameters['run_time'] = run_time
		
		json.dump(general_parameters, outfile)	
		
		outfile.close()
		print('\n Best model achieves', best_search[0], 'with parameters', best_search[1], '... yay! \n')
		
		
	def updateParameters(self, best):

		for key in best:
			if 'learning_rate' in key:
				self.learning_rate = best[key]
			elif 'momentum' in key:
				self.momentum = best[key]
			elif 'init_mode' in key:
				self.init_mode = best[key]		
			elif 'activation1' in key:
				self.activation1 = best[key]		
			elif 'activation2' in key:
				self.activation2 = best[key]						
			elif 'dropout_rate' in key:
				self.dropout_rate = best[key]		
			elif 'weight_constraint' in key:
				self.weight_constraint = best[key]		
			elif 'hidden_size' in key:
				self.hidden_size = best[key]		
			elif 'loss' in key:
				self.loss = best[key]		
			elif 'optimiser' in key:
				self.optimiser = best[key]		
			elif 'epochs' in key:
				self.epochs = best[key]		
			elif 'batch_size' in key:
				self.batch_size = best[key]	
			elif 'model_string' in key:
				self.model_string = best[key]	
			elif 'layers' in key:
				self.layers = best[key]									
					

def defineDL(parameters):
		
	models = ['LSTM', 'RNN', 'GRU', 'MLP']
	dl = ""
		
	st = datetime.datetime.fromtimestamp(time.time()).strftime('%Y-%m-%d-%H:%M:%S')		
		
	if not 'model_string' in parameters:
		print("Error - you need to provide a model_string out of: ", models)
	else:
		model_string = parameters.get('model_string', 'MLP')	
		layers = parameters.get('layers', 2)
		batch_size = parameters.get('batch_size', 32)
		epochs = parameters.get('epochs', 20)
		dropout_rate = parameters.get('dropout_rate', 0)
		weight_constraint = parameters.get('weight_constraint', 0)		
		hidden_size = parameters.get('hidden_size', 20)
		embedding_size = parameters.get('embedding_size', 10000)
		learning_rate = parameters.get('learning_rate', 0.01)
		momentum = parameters.get('momentum', 0.9)
		optimiser = parameters.get('optimiser', 'adam')
		init_mode = parameters.get('init_mode', 'uniform')		
		activation1 = parameters.get('activation1', 'relu')				
		activation2 = parameters.get('activation2', 'sigmoid')		
		loss = parameters.get('loss', 'categorical_crossentropy')
		eval_metrics = parameters.get('eval_metrics', ['accuracy'])		
		model_name = parameters.get('model_name', 'model_'+st)
		prediction = parameters.get('prediction', 'regression')		
		
		
		dl = DeepLearner(model_string, layers, batch_size, epochs, dropout_rate, weight_constraint, hidden_size, embedding_size, learning_rate, momentum, optimiser, init_mode, activation1, activation2, loss, eval_metrics, model_name, prediction)
	return dl

	
