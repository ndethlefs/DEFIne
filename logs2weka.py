import os
import json

print('Reading Viper logs and preparing Weka files...')

path = "./logs/"
weka_file = 'weka_out.arff'
legend = '@attribute name {names} \n' + '@attribute search {searches} \n' + '@attribute init_mode {init_modes} \n' + '@attribute loss {losses} \n' + '@attribute modelString {models} \n' + '@attribute epochs {epochs} \n' + '@attribute learning_rate {learning_rates} \n' + '@attribute batch_size {batch_sizes} \n' + '@attribute dropout_rate {dropout_rates} \n' + '@attribute hidden_size NUMERIC \n' + '@attribute layers NUMERIC \n' + '@attribute optimiser {optimisers} \n' + '@attribute momentum {momentums} \n' + '@attribute activation1 {activation1s} \n' + '@attribute activation2 {activation2s} \n' + '@attribute weight_constraint {weight_constraints} \n' + '@attribute fit_time NUMERIC \n' + '@attribute cpu_cores NUMERIC \n' + '@attribute search_algorithm {grid, random} \n' + '@attribute search_space NUMERIC \n' + '@attribute hardware {cpu, gpu, unknown} \n'  + '@attribute score NUMERIC \n' 

names = []
searches = []
init_modes = []
losses = []
models = []
epochs = []
learning_rates = []
batch_sizes = [] 
dropout_rates = []
hidden_sizes = []
layers = []
optimisers = []
momentums = []
activation1s = []
activation2s = []
weight_constraints = []
hardwares = []



def loadJson(filename):

	json_data=open(filename).read()
	data = json.loads(json_data)
	return data

def writeWeka(dict, hardware, name):

	print("Traversing searches...")
	weka_entries = []
	
	search = 1
	
	while search<=100:	
		search_no = 'search_'+ str(search)
		if search_no in dict:
			weka_line = [name]
			names.append(name)
			weka_line.append(search_no)						
			searches.append(search_no)
			weka_line.append(dict[search_no]["param"]["init_mode"])
			init_modes.append(dict[search_no]["param"]["init_mode"])
			weka_line.append(dict[search_no]["param"]["loss"])
			losses.append(dict[search_no]["param"]["loss"])		
			weka_line.append(dict[search_no]["param"]["modelString"])
			models.append(dict[search_no]["param"]["modelString"])		
			weka_line.append(dict[search_no]["param"]["epochs"])
			epochs.append(dict[search_no]["param"]["epochs"])		
			weka_line.append(dict[search_no]["param"]["learning_rate"])
			learning_rates.append(dict[search_no]["param"]["learning_rate"])		
			weka_line.append(dict[search_no]["param"]["batch_size"])
			batch_sizes.append(dict[search_no]["param"]["batch_size"])		
			weka_line.append(dict[search_no]["param"]["dropout_rate"])
			dropout_rates.append(dict[search_no]["param"]["dropout_rate"])				
			weka_line.append(dict[search_no]["param"]["hidden_size"])
			hidden_sizes.append(dict[search_no]["param"]["hidden_size"])		
			weka_line.append(dict[search_no]["param"]["layers"])
			layers.append(dict[search_no]["param"]["layers"])		
			weka_line.append(dict[search_no]["param"]["optimiser"])
			optimisers.append(dict[search_no]["param"]["optimiser"])		
			weka_line.append(dict[search_no]["param"]["momentum"])
			momentums.append(dict[search_no]["param"]["momentum"])		
			weka_line.append(dict[search_no]["param"]["activation1"])
			activation1s.append(dict[search_no]["param"]["activation1"])		
			weka_line.append(dict[search_no]["param"]["activation2"])
			activation2s.append(dict[search_no]["param"]["activation2"])		
			weka_line.append(dict[search_no]["param"]["weight_constraint"])		
			weight_constraints.append(dict[search_no]["param"]["weight_constraint"])				
			weka_line.append(dict[search_no]["fit_time"])
			weka_line.append(dict["cpu_cores"])
			weka_line.append(dict["search_algorithm"])
			weka_line.append(dict["search_space"])		
			if hardware=='cpu':
				weka_line.append('cpu')
				hardwares.append('cpu')	
			elif hardware=='gpu':
				weka_line.append('gpu')			
				hardwares.append('cpu')	
			else:
				weka_line.append('unknown')	
				hardwares.append('cpu')	
			weka_line.append(dict[search_no]["score"])
		
#		print(','.join(weka_line))
			print(str(weka_line))
			weka_entries.append(weka_line)
		search = search + 1 	
	return weka_entries	
	

write_file = open(weka_file, 'w')
write_file.writelines('@relation hyperparameters.symbolic \n \n')
write_file.writelines('### copy the legend here... \n \n')
write_file.writelines('@data \n')
for file in os.listdir(path):
	if file.endswith('.json'):
		print(file)
		dict = loadJson(path+file)
		print(dict["search_2"]["param"]["activation1"])
		hardware = 'unknown'
		if 'cpu' in file:
			hardware = 'cpu'
		elif 'gpu' in file:
			hardware = 'gpu'	
		print(file)
		weka_entries = writeWeka(dict, hardware, file.split('.json')[0])
		for w in weka_entries:
			w = str(w).replace('[', '').replace(']', '').replace(' ', '').replace("'", '')
			write_file.writelines(w+'\n')


legend = legend.replace('{names}', str(set(names)).replace('[', '').replace(']', '').replace("'", ''))
legend = legend.replace('{losses}', str(set(losses)).replace('[', '').replace(']', '').replace("'", ''))
legend = legend.replace('{searches}', str(set(searches)).replace('[', '').replace(']', '').replace("'", ''))
legend = legend.replace('{init_modes}', str(set(init_modes)).replace('[', '').replace(']', '').replace("'", ''))
legend = legend.replace('{models}', str(set(models)).replace('[', '').replace(']', '').replace("'", ''))
legend = legend.replace('{learning_rates}', str(set(learning_rates)).replace('[', '').replace(']', '').replace("'", ''))
legend = legend.replace('{dropout_rates}', str(set(dropout_rates)).replace('[', '').replace(']', '').replace("'", ''))
legend = legend.replace('{batch_sizes}', str(set(batch_sizes)).replace('[', '').replace(']', '').replace("'", ''))
legend = legend.replace('{momentums}', str(set(momentums)).replace('[', '').replace(']', '').replace("'", ''))
legend = legend.replace('{activation1s}', str(set(activation1s)).replace('[', '').replace(']', '').replace("'", ''))
legend = legend.replace('{activation2s}', str(set(activation2s)).replace('[', '').replace(']', '').replace("'", ''))
legend = legend.replace('{weight_constraints}', str(set(weight_constraints)).replace('[', '').replace(']', '').replace("'", ''))
legend = legend.replace('{epochs}', str(set(epochs)).replace('[', '').replace(']', '').replace("'", ''))
legend = legend.replace('{optimisers}', str(set(optimisers)).replace('[', '').replace(']', '').replace("'", ''))

write_file.writelines('\n' + legend + '\n')

		
write_file.close()






