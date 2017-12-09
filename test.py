# load or pre-process dataset as X, Y
data = DataSet(X_set, Y_set)
data.representData().shuffleAndSplit()

parameters = {
'model_string' : 'MLP',
'layers' : 2,
'hidden_size' : 10
}

dl = defineDL(parameters)
dl.designModel(data).compileModel().trainModel(data)