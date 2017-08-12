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
		
		
		# code for abalone
		
		if 'abalone' in name:
			print("Abalone dataset predicting the age in integers - can be a classification or regression task. Now treating it as a classification task and treating outputs as strings.")								
			for line in open(self.repository_path + 'abalone-classification.txt', 'r'):
				l = line.split(',')[:-1]
				first, rest = l[0], l[1:]
				f = 0
				r = []
				if first=='M':
					first = 0
				elif first=='F':
					first = 1
				else:
					first = 2
				for element in rest:
					r.append(int(float(element)))
				X_set.append([f]+r)			
				
				Y_set.append(int(line.split(',')[-1].replace('\n', '')))
		
		# code for adult (test and train)
		
		if 'adult' in name:
			print("Adult dataset predicting someone's income as more or less than 50K dollars based on socioeconomic features.")	
			print('Removing feature fnlwgt (continuous) - these are final sampling weights that (I guess?) are mostly needed for other classifiers...')
			for line in open(self.repository_path + 'adult-training.txt', 'r'):
				data = line.split(',')[:-1]
				x = []
				third = data[2]
				data.remove(third)
				for i in data:
					x.append(str(i).strip())
				X_set.append(x)	
				label = line.split(',')[-1]
				Y_set.append(str(label).replace('\n', '').strip())	
			
			for line in open(self.repository_path + 'adult-test.txt', 'r'):
				data = line.split(',')[:-1]
				x = []
				third = data[2]
				data.remove(third)
				for i in data:
					x.append(str(i).strip())
				X_set.append(x)	
				label = line.split(',')[-1]
				Y_set.append(str(label).replace('\n', '').strip())				
			
		
		# code for bank-full
		
		if 'bank' in name:
			print("Bank dataset predicting on whether based on an advert someone will open an account (1) or not (0).")
			print("For outputs, merging categories y and yes.")
			for line in open(self.repository_path + 'bank-full.txt', 'r'):
				l = line.replace('"', '')
				data = l.split(',')[:-1]			
				X_set.append(data)	
				label = l.split(',')[-1].replace('\n', '')
				if label=='y' or label=='yes':
					Y_set.append(1)
				else:	
					Y_set.append(0)
		
		# code for breast_cancer_wisconsin
		
		if 'breast-cancer' in name:
			print("Wisconsin breast cancer data set: diagnosis from medical data, outputs are B=benign, M=malignant, converted to 0 (benign) and 1 (malignant).")		
			for line in open(self.repository_path + 'breast-cancer-wisconsin.txt', 'r'):
				data = line.split(',')[1:-1]
				label = int(line.split(',')[-1].replace('\n', ''))
				x = []
				for i in data:
					if i=='?':
						x.append(100)
					else:
						x.append(int(i))
				X_set.append(x)
				print(label)
				if label==2:
					Y_set.append(0)
				else:
					Y_set.append(1)
	
		
		# code for car
		
		if 'car' in name:
			print("Car datasets classifying cars as acceptable or unacceptable based on a set of car attributes.")								
			for line in open(self.repository_path + 'car.txt', 'r'):
				X_set.append(line.split(',')[:-1])
				Y_set.append(line.split(',')[-1].replace('\n', ''))			
		
		# code for forestfires
		
		if 'forestfires' in name:
			print("Forest fires dataset: this is a regression task, output is the area of forest burnt.")								
			for line in open(self.repository_path + 'forestfires.txt', 'r'):
				X_set.append(line.split(',')[:-1])
				label = line.split(',')[-1].replace('\n', '')				
				Y_set.append(float(label))
		
		# code for iris.data
		
		if 'iris' in name:
			print("Iris dataset, output is one of three types if Iris, range 1-3.")		
			for line in open(self.repository_path + 'iris.data2.txt', 'r'):
				data = line.split(',')[:-1]
				label = line.split(',')[-1].replace('\n', '')
				x = []
				for i in data:
					x.append(float(i))
				X_set.append(x)
				l = int(label)
				Y_set.append(l)
	
		
		# code for poker-hand (training and testing)
		
		if 'poker' in name:
			print("Poker hand classification: output is a hand out of 0-9")						
			for line in open(self.repository_path + 'poker-hand-training-true.txt', 'r'):
				data = line.split(',')[:-1]
				label = line.split(',')[-1]
				x = []
				for i in data:
					x.append(int(i))
				X_set.append(x)
				l = int(label)
				Y_set.append(int(label))
			for line in open(self.repository_path + 'poker-hand-testing.txt', 'r'):
				data = line.split(',')[:-1]
				label = line.split(',')[-1]
				x = []
				for i in data:
					x.append(int(i))
				X_set.append(x)
				l = int(label)
				Y_set.append(int(label))			
		
		# code for UCI_HAR (train and test)						
		
		if 'uci_har' in name:
			print("Human activity detection from smartphones: features are sensor data, outputs is an activity label (out of 6).")				
			print('Using unity-based normalisation to rescale all numbers to be between 0 and 1.')
			for line in open(self.repository_path + 'UCI_HAR_train.txt', 'r'):
				data = line.split(',')[:-1]
				label = line.split(',')[-1]
				x = []
				for i in data:
					x.append(float(i))
				X_set.append(x)
				l = int(label)
				Y_set.append(int(label))
			for line in open(self.repository_path + 'UCI_HAR_test.txt', 'r'):
				data = line.split(',')[:-1]
				label = line.split(',')[-1]
				x = []
				for i in data:
					x.append(float(i))
				X_set.append(x)
				l = int(label)
				Y_set.append(int(label))												
			X_set = normalize(X_set)	
		
		# code for wine
		
		if 'wine' in name:
			print("Wine dataset: features are chemical analysis of wines, output is one of three regions (output is first attribute in input vector).")		
			for line in open(self.repository_path + 'wine.txt', 'r'):
				data = line.split(',')[1:-1]
				label = line.split(',')[0].replace('\n', '')
				x = []
				for i in data:
					x.append(float(i))
				x.append(float(line.split(',')[-1])/10)	
				X_set.append(x)
				l = int(label)
				Y_set.append(l)		
		
		# code for winequality-red
		
		if 'winequality-red' in name:
			print("Wine quality dataset (red): features are chemical and outputs are out 10.")
			for line in open(self.repository_path + 'winequality-red.txt', 'r'):
				data = line.split(',')[:-1]
				label = line.split(',')[-1]
				x = []
				for i in data:
					x.append(float(i))
				X_set.append(x)
				l = int(label)
				Y_set.append(int(label))
		
		# code for winequality-white
		
		if 'winequality-white' in name:
			print("Wine quality dataset (white): features are chemical and outputs are out 10.")		
			for line in open(self.repository_path + 'winequality-white.txt', 'r'):
				data = line.split(',')[:-1]
				label = line.split(',')[-1]
				x = []
				for i in data:
					x.append(float(i))
				X_set.append(x)
				l = int(label)
				Y_set.append(int(label))
		
		# code for processed-cleveland.data-out								

		if 'cleveland' in name:
			print("Cleveland heart disease dataset: classification from medical features, outputs are 0 or 1.")
			for line in open(self.repository_path + 'processed.cleveland.data-out.txt', 'r'):
				data = line.split(',')[:-1]
				print('data', data)
				label = line.split(',')[-1]
				x = []
				for i in data:
					x.append(float(i))
				X_set.append(x)
				l = int(label)
				if l==0:
					Y_set.append(0)
				else:
					Y_set.append(1)
					
		else:
			print('Any other dataset, attempting to determine features...')
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


			
		
#print('testing...')
#dataset = DataRepository()
#dataset.load('uci_har')
#print(dataset.X_set)
#print(dataset.Y_set)








