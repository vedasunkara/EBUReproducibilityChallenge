import matplotlib.pyplot as plt

import numpy as np

import os

directory_name = "DATA/EBU=True,RAND=False,WALL=0.5/" #"C:/Users/zach_surf/Desktop/rl/qlearning/MNISTMAZE/dqn_20_percent/" #"DATA/" #"C:/Users/zach_surf/Desktop/rl/qlearning/MNISTMAZE/dqn_20_percent/" # 
directory = os.fsencode(directory_name)





averaged_data = []




for file in os.listdir(directory):
		filename = os.fsdecode(file)
		data = np.load(directory_name + filename) # np.load('dqn_20_percent/'+str(t)+"_results.npy")
		# rel_lengths = [x[0] for x in data]
		# steps = steps = [x[1] for x in data]

		# print(steps)
		#plt.plot(data[:,0],data[:,1])

		data_in_buckets = np.zeros((len(range(0,200000,2000))))
		for x in range(0,200000,2000):
			total = 0
			for i,d in enumerate(data[:,0]):
				if d >= x and d < x + 2000:
					data_in_buckets[int(x/2000)] += data[i][1]
					total +=1

			#if total > 0:
			data_in_buckets[int(x/2000)] /= total


		averaged_data.append(data_in_buckets)


		#exit()



averaged_data =np.stack(averaged_data)

mean = np.mean(averaged_data,axis=0)
std = np.std(averaged_data,axis=0)

print(np.mean(mean*18))

plt.plot(range(0,200000,2000),mean+1.96*std/np.sqrt(len(averaged_data))) #15
plt.plot(range(0,200000,2000),mean-1.96*std/np.sqrt(len(averaged_data))) 
plt.plot(range(0,200000,2000),mean) #15



plt.show()
