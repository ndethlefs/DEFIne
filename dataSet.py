import numpy as np
from keras.preprocessing import sequence
from keras.engine.training import _slice_arrays
import numbers
from keras.utils import np_utils
import sys
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from sklearn.decomposition import PCA


class DataSet:

	def __init__(self, X_set, Y_set, prediction='classification'):
		self.X_set = X_set
		self.Y_set = Y_set		
		self.max_len = 0

		self.X_train = []
		self.Y_val = []
		self.Y_train = []
		self.Y_val = []
	
		self.text = 'mask_zeros '
		self.vocab = set()		  
		self.char_indices = {}
		self.indices_char = {}
		self.dimensionality = ''
		self.outputs = 1		
		if not hasattr(Y_set[0], '__iter__'):
			self.outputs = len(set(Y_set))			
		self.prediction = prediction
			
		# determine number of symbols and maximum sequence length.
		self.countSymbols()
		self.countMaxLen()			
		
		# keep a copy of the flat output label vector for visualisation.
		self.labels = Y_set



	def representData(self):
		# Find out what representation is needed, 2D or 3D. 	
		sample_X = self.X_set[0]		
		
		if self.prediction=='classification':
			labels = set(self.Y_set)
			label_mappings = {}
			for i, item in enumerate(labels):
				label_mappings[item] = int(i)
				
			for j, jtem in enumerate(self.Y_set):
				self.Y_set[j] = label_mappings[jtem]

		self.Y_set = np.asarray(self.Y_set, dtype=np.int32)
		sample_Y = self.Y_set[0] 
		self.outputs = len(set(self.Y_set))
				
		if hasattr(sample_Y, '__iter__'):
			print('Use a 3D representation to model sequence-to-sequence task... (encoding.)')		
			self.dimensionality = '3D'			
			X, Y = self.representDataAs3D(self.X_set, self.Y_set)
		else:
			self.dimensionality ='2D'		
			types = [isinstance(n, numbers.Number) for n in self.X_set[0]]
			if not False in types:
				print('Use a 2D representation for classification task (numeric values)... (no encoding.)')			
				X, Y = self.representDataAs2DNumeric(self.X_set, self.Y_set)	
			else:
				print('Use a 2D representation for classification task (symbolic values)... (encoding.)')						
				X, Y = self.representDataAs2D(self.X_set, self.Y_set)	

		self.X_set = X
		self.Y_set = Y		

		return self


	def representDataAs2DNumeric(self, X_set, Y_set):
		# Use this for tasks that classify a single output value.
		X_2D = np.asarray(X_set)
		Y_2D = np.asarray(Y_set)		

		new_X = X_2D
		# Look into this later - paddings pads them all to 0s and 1s...
#		new_X = sequence.pad_sequences(X_2D, maxlen=self.max_len, value=0)
		
		if self.prediction=='classification':			
			new_Y = np_utils.to_categorical(Y_2D, self.outputs)				
		else:
			new_Y = Y_2D		
	
		print('Representing data X with shape', new_X.shape,  'and Y with shape:', new_Y.shape)	
		
		return new_X, new_Y


	def representDataAs2D(self, X_set, Y_set):
		# Use this for tasks that classify a single output value.
		X_2D = np.asarray(X_set)
		Y_2D = np.asarray(Y_set)		
		
		# Get mapping dicts for input representations.
		self.char_indices = dict((c, i) for i, c in enumerate(self.vocab))
		self.indices_char = dict((i, c) for i, c in enumerate(self.vocab)) 
		#print self.char_indices		
		
		new_X = self.encode2DIntegerX(X_2D)
		new_X = sequence.pad_sequences(new_X, maxlen=self.max_len)		
	
		if self.prediction=='classification':
			new_Y = np_utils.to_categorical(Y_2D, self.outputs)		
		else:
			new_Y = Y_2D								
			
		print('Representing data X with shape', new_X.shape,  'and Y with shape:', new_Y.shape)			
				
		return new_X, new_Y
		
		
	def representDataAs3D(self, X_set, Y_set):
		# Use this for sequence-to-sequence learning with 3D methods below.		
		X_3D = np.zeros((len(self.X_set), self.max_len, len(self.vocab)), dtype=np.bool)
		Y_3D = np.zeros((len(self.Y_set), self.max_len, len(self.vocab)), dtype=np.bool)		
		
		# Get mapping dicts for input representations.
		self.char_indices = dict((c, i) for i, c in enumerate(self.vocab))
		self.indices_char = dict((i, c) for i, c in enumerate(self.vocab))  				
		
		print('Representing data X and Y with shape:', X_3D.shape)
		
		new_X = self.encode3DBooleanX(X_set)
		new_Y = self.encode3DBooleanX(Y_set)	
		print('Padding all sequences to max_len:', self.max_len)
		new_X = sequence.pad_sequences(new_X, maxlen=self.max_len)
		new_Y = sequence.pad_sequences(new_Y, maxlen=self.max_len)		
		
		return new_X, new_Y
		

	def encode2DIntegerX(self, X_set):		
		# encode X into a 2D matrix / array of arrays.
		for i, item in enumerate(X_set):
			for j, jtem in enumerate(item):
				item[j] = self.char_indices[jtem]
			X_set[i] = item
		return X_set	
			

	def encode3DBooleanX(self, X_set):
		# encode X into a boolean matrix
		X_3D = np.zeros((len(self.X_set), self.max_len, len(self.vocab)), dtype=np.bool)
		for i, item in enumerate(X_set):
			X1 = np.zeros((self.max_len, len(self.vocab)))
			for j, c in enumerate(item):
				X1[j, self.char_indices[c]] = 1
			X_3D[i] = X1			
		return X_3D	


	def encode3DBooleanY(self, Y_set):
		# encode Y into a boolean matrix
		for i, item in enumerate(Y_set):
			Y1 = np.zeros((self.max_len, len(self.vocab)))
			for j, c in enumerate(item):
				Y1[j, char_indices[c]] = 1
			Y[i] = Y1	
		return Y_set
	
	
	
	def decode3D(self, X, calc_argmax=True):
		# this decodes 3D items. 
		if calc_argmax:
			X = X.argmax(axis=-1)
		return ' '.join(indices_char[x] for x in X)	
		
	
	def decode2D(X_item, model):
		# this decodes 2D X items. 
		new_X = []
		for x in X_item:
			new = indices_char[int(x)]
			new_X.append(new)
		new_X = ' '.join(new_X).replace('mask_zeros', '')
		return ' '.join(new_X.split())		
		
		
	def countSymbols(self):
		# find all the different types of symbols in the input and output. 
		
		types = [isinstance(n, numbers.Number) for n in self.X_set[0]]
		if False in types:		
			self.vocab = self.assembleVocab()
		else:
			print('Using numeric values only - no vocabulary needed.')
			self.vocab = set()
			

	def assembleVocab(self):
	
		for x in self.X_set:
			self.text = self.text + ' '.join(x) + ' '

		if hasattr(self.Y_set[0], '__iter__'):
			for y in self.Y_set:
				self.text = self.text + ' '.join(y) + ' '
		
		chars = set(self.text.split())
		vocab = list(chars)
		vocab.insert(0, 'mask_zeros')		
		print('Found', len(vocab), 'unique vocabulary items.')
			
		return vocab


	def countMaxLen(self):
		# find the max len of an input or output sequence so we can pad all sequences. 
		for x in self.X_set:
			if len(x) > self.max_len:
				self.max_len = len(x)
		if hasattr(self.Y_set[0], '__iter__'):				
			for y in self.Y_set:
				if len(y) > self.max_len:
					self.max_len = len(y)		

		
	def shuffleAndSplit(self):		
		
		# Shuffle (X, y) in unison as the later parts of X will almost all be larger digits	
		indices = np.arange(len(self.Y_set))
		X = self.X_set[indices]
		Y = self.Y_set[indices]
		
		# Explicitly set apart 10% for validation data that we never train over		
		split_at = int(len(X) * 0.8)
		print(split_at)
		X_train = X[:split_at]
		X_val = X[split_at:]
		Y_train = Y[:split_at]		
		Y_val = Y[split_at:]				
		
#		(X_train, X_val) = (_slice_arrays(X, 0, split_at), _slice_arrays(X, split_at))
#		(Y_train, Y_val) = (Y[:int(split_at)], Y[int(split_at):])		
		
		print('Shuffled and split dataset into', len(X_train), 'training instances and', len(X_val), 'test instances.')
		
		self.X_train = X_train
		self.Y_train = Y_train
		self.X_val = X_val
		self.Y_val = Y_val						
		
		return X_train, X_val, Y_train, Y_val
			
			
	def visualise(self):
	
		X = self.X_set
		y = self.labels

		fig = plt.figure(0, figsize=(10, 8))
		ax = Axes3D(fig, rect=[0, 0, .95, 1], elev=48, azim=134)

		pca = PCA(n_components=3)
		X_reduced = pca.fit_transform(X)
		used_labels = []

		colour_scheme = plt.cm.Set1

		for i, item in enumerate(X_reduced):
			if not y[i] in used_labels:
				ax.scatter(item[0], item[1], item[2], c = colour_scheme(y[i]/10), s=40, edgecolor='k', label=y[i])
				used_labels.append(y[i])
			else:
				ax.scatter(item[0], item[1], item[2], c = colour_scheme(y[i]/10), s=40, edgecolor='k')	
		

		ax.legend(loc="lower right")           
		ax.set_title("First three PCA directions")
		ax.set_xlabel("1st eigenvector")
		ax.w_xaxis.set_ticklabels([])
		ax.set_ylabel("2nd eigenvector")
		ax.w_yaxis.set_ticklabels([])
		ax.set_zlabel("3rd eigenvector")
		ax.w_zaxis.set_ticklabels([])
	
		plt.show()
	
			
		
		
