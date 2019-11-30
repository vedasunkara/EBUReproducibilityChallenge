import matplotlib.pyplot as plt

import numpy as np

import os

directory_name = "DATA/"
directory = os.fsencode(directory_name)


graph_data = {}



for file in os.listdir(directory):
		filename = os.fsdecode(file)
		if filename[0:4] !="EBU=":
			continue
		terms = filename.split(",")
		walls = terms[3].split("=")[-1]
		random = terms[2].split("=")[-1]
		beta = None
		if "EBU=True" in filename:
				of_type = "EBU" 
				beta = terms[-1].split("_")[0]
		elif "NSTEP=True" in filename:
				of_type = "NSTEP DQN"
				print(filename)
		else:
				of_type = "DQN"

		graph_name = str((walls,random))
		
		if graph_name not in graph_data: graph_data[graph_name] = {}

		model_type = str((of_type,beta))

		if model_type not in graph_data[graph_name]: graph_data[graph_name][model_type] = []

		graph_data[graph_name][model_type].append(np.load(directory_name + filename)) 


def process_trial(data):
		data_in_buckets = np.zeros((len(range(0,200000,2000))))
		for x in range(0,200000,2000):
			total = 0
			for i,d in enumerate(data[:,0]):
				if d >= x and d < x + 2000:
					data_in_buckets[int(x/2000)] += data[i][1]
					total +=1
			data_in_buckets[int(x/2000)] /= total

		return data_in_buckets



fig, axes = plt.subplots(1, 3,figsize=(10,3))
fig.suptitle('MNISTMAZE Trials')

line_format = {
str(('EBU', 'BETA=1.0')):("red","solid"),
str(('DQN', None)):("green","dotted"),
str(('NSTEP DQN', None)):("blue","dashed"),
str(('EBU', 'BETA=0.25')):("purple","solid"),
str(('EBU', 'BETA=0.5')):("blue","solid"),
str(('EBU', 'BETA=0.75')):("yellow","solid")
}

name_map = {
str(('EBU', 'BETA=1.0')):"EBU (BETA=1.0)",
str(('DQN', None)):"One-Step DQN",
str(('NSTEP DQN', None)):"N-Step DQN",
str(('EBU', 'BETA=0.25')):"EBU (BETA=0.25)",
str(('EBU', 'BETA=0.5')):"EBU (BETA=0.5)",
str(('EBU', 'BETA=0.75')):"EBU (BETA=0.75)"
}


axes[0].title.set_text('Deterministic, Wall density: 20%')
axes[1].title.set_text('Deterministic, Wall density: 50%')
axes[2].title.set_text('Stochastic, Wall density: 50%')



for i,g in enumerate(graph_data.keys()):
	print(g)
	for k in graph_data[g].keys():
		print(k)
		averaged_data = []
		color,style = line_format[str(k)]
		for trial in graph_data[g][k]:
				averaged_data.append(process_trial(trial))
		averaged_data =np.stack(averaged_data)
		axes[i].plot(range(0,200000,2000),np.median(averaged_data,axis=0),color=color,linestyle=style,label=name_map[str(k)]) #15
		axes[i].set_ylabel("Relative Lengths")
		axes[i].set_xlabel("Steps")
		axes[i].legend()


plt.show()
		


# mean = np.mean(averaged_data,axis=0)
# std = np.std(averaged_data,axis=0)

# print(np.mean(mean*18))

# # plt.plot(range(0,200000,2000),mean+1.96*std/np.sqrt(len(averaged_data))) #15
# # plt.plot(range(0,200000,2000),mean-1.96*std/np.sqrt(len(averaged_data))) 
# plt.plot(range(0,200000,2000),np.median(averaged_data,axis=0)) #15



