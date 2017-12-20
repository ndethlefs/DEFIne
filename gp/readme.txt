gp_data contains outputs of hyperparameter optimisations over various source datasets from UCI repository.

response.py will run hyperparameter optimisation over those output files to fit Gaussian Processes and identify the most predictive hyperparameters for each of these datasets. It does this by analysing the length scales.

analyse_response.py produces a json file that represents each dataset as a vector of metrics about the dataset and contains the “best” hyperparameters.