import numpy as np
from keras.preprocessing import sequence
from keras.engine.training import _slice_arrays
import numbers
from keras.utils import np_utils
import sys
from sklearn.preprocessing import *

class DataRepository:

	def __init__(self):
		self.repository_path = './data_repository/'
		self.X_set = []
		self.Y_set = []

	
	def isFloat(self, value):
		try:
			float(value)
			return True
		except ValueError:
			return False	

	def isInt(self, value):
		try:
			int(value)
			return True
		except ValueError:
			return False	

	def load(self, name):
		
		X_set = []
		Y_set = []		
		
		
		print('Loading dataset, attempting to determine features...')
		for line in open(self.repository_path + name, 'r'):	
			data = line.split(',')[:-1]
			x = []
			for i in data:
				if self.isInt(i):
					x.append(int(i))
				elif self.isFloat(i):
					x.append(float(i))						
				else:
					x.append(i)	
			X_set.append(x)
			label = line.split(',')[-1].replace('\n', '')
			if self.isInt(label):
				Y_set.append(int(label))				
			elif self.isFloat(label):
				Y_set.append(float(label))									
			else:
				Y_set.append(label)		

	self.X_set = np.asarray(X_set)
	self.Y_set = np.asarray(Y_set)		
	return self.X_set, self.Y_set
