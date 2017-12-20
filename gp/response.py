print(__doc__)

'''
Using scikit learn's Gaussian Process regressor and random search hyperparameter optimisation to find the identify 
length scales from experiments on source datasets and find the most predictive hyperparameters for source datasets.
Nina, 20 Dec 2017.
'''

import numpy as np
import pandas
import sys
import os
from matplotlib import pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV

from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, ConstantKernel as C
from sklearn.metrics import mean_squared_error, mean_absolute_error

np.random.seed(1)
		
features = {
'name' : 0,
'search' : 1,
'init_mode' : 2,
'loss' : 3,
'modelString' : 4,
'epochs' : 5,
'learning_rate' : 6,
'batch_size' : 7,
'dropout_rate' : 8,
'hidden_size' : 9,
'layers' : 10,
'optimiser' : 11,
'momentum' : 12,
'activation1' : 13,
'activation2' : 14,
'weight_constraint' : 15,
'fit_time' : 16,
'cpu_cores' : 17,
'search_algorithm' : 18,
'search_space' : 19,
'hardware' : 20,
'learning_tasks' : 21,
'size_of_datasets' : 22,
'input_features' : 23,
'output_features' : 24,
'data_type' : 25,
'vocab_size' : 26,
'dimensionality' : 27,
'score' : 28,
}		


def loadData(file, feature):

	infile = open('./gp_data_regression/bike_sharing_day.txt', 'r')
	X_set = []
	Y_set = []		
	for line in infile:
		data = line.replace("\n", "").split(",")
#	x_data = data[1:-1]
		x_data = data[features[feature]]
#	new_x = []
#	for x in x_data:
#		new_x.append(float(x))		
		X_set.append(float(x_data))		
		label = data[-1]	
		Y_set.append(float(label))

	infile.close()
	return X_set, Y_set

X = []
Y = []

def create_gp(kernel=C(1.0, (1e-3, 1e3)) * RBF(10, (1e-2, 1e2))):
	gp1 = GaussianProcessRegressor(kernel=kernel, n_restarts_optimizer=9)
	return gp1
	
results_file = open("results_regression.txt", 'w')	

output = []

for f in os.listdir("./gp_data_regression/"):
	print(f)
	if f.endswith(".txt"):	
		print("Reading dataset", f)
		output.append(str(f))
		output.append("================================")
		for fea in features:
			if not fea=="name":
				X, Y = loadData(f, fea)
				X = np.asarray(X)
				X = np.atleast_2d(X).T
				y = np.asarray(Y)
				print(X, y)
				output.append(X)
				output.append(y)
		
				gp1 = create_gp(kernel=C(1.0, (1e-3, 1e3)) * RBF(10, (1e-2, 1e2)))

				param_grid = dict(kernel=[C(1.0, (1e-3, 1e3)) * RBF(1.0, (1e-2, 1e2)), C(1.0, (1e-3, 1e3)) * RBF(2.0, (1e-2, 1e2)),  C(1.0, (1e-3, 1e3)) * RBF(3.0, (1e-2, 1e2)), C(1.0, (1e-3, 1e3)) * RBF(4.0, (1e-2, 1e2)), C(1.0, (1e-3, 1e3)) * RBF(5.0, (1e-2, 1e2)), C(1.0, (1e-3, 1e3)) * RBF(6.0, (1e-2, 1e2)), C(1.0, (1e-3, 1e3)) * RBF(7.0, (1e-2, 1e2)), C(1.0, (1e-3, 1e3)) * RBF(8.0, (1e-2, 1e2)), C(1.0, (1e-3, 1e3)) * RBF(9.0, (1e-2, 1e2)), C(1.0, (1e-3, 1e3)) * RBF(10.0, (1e-2, 1e2))])
				grid = GridSearchCV(estimator=gp1, param_grid=param_grid, n_jobs=1, cv=2)
				grid_result = grid.fit(X, y)
				means = grid_result.cv_results_['mean_test_score']
				stds = grid_result.cv_results_['std_test_score']
				params = grid_result.cv_results_['params']
				print("Best: %f using %s" % (grid_result.best_score_, grid_result.best_params_))
				output.append("Best: %f using %s" % (grid_result.best_score_, grid_result.best_params_))
				for mean, stdev, param in zip(means, stds, params):
					print("%f (%f) with: %r" % (mean, stdev, param))
				print('beeest', grid_result.best_params_['kernel'])		
				output.append('beeest' + str(grid_result.best_params_['kernel']))
				output.append("================================")
	for o in output:
		print(o)
		results_file.writelines(str(o) + "\n")
	output = []


results_file.close()

def f(x):
    """The function to predict."""
    return x * np.sin(x)


# Mesh the input space for evaluations of the real function, the prediction and
# its MSE
x = np.atleast_2d(np.linspace(0, 10, 1000)).T

# Instanciate a Gaussian Process model

kernel=C(1.0, (1e-3, 1e3)) * RBF(10.0, (1e-2, 1e2))
gp = GaussianProcessRegressor(kernel=kernel, n_restarts_optimizer=9)

# Fit to data using Maximum Likelihood Estimation of the parameters
gp.fit(X, y)

# Make the prediction on the meshed x-axis (ask for MSE as well)
y_pred, sigma = gp.predict(x, return_std=True)

# Plot the function, the prediction and the 95% confidence interval based on
# the MSE
fig = plt.figure()
plt.plot(x, f(x), 'r:', label=u'$f(x) = x\,\sin(x)$')
plt.plot(X, y, 'r.', markersize=10, label=u'Observations')
plt.plot(x, y_pred, 'b-', label=u'Prediction')
plt.fill(np.concatenate([x, x[::-1]]),
         np.concatenate([y_pred - 1.9600 * sigma,
                        (y_pred + 1.9600 * sigma)[::-1]]),
         alpha=.5, fc='b', ec='None', label='95% confidence interval')
plt.xlabel('$x$')
plt.ylabel('$f(x)$')
plt.ylim(-10, 20)
plt.legend(loc='upper left')

plt.show()