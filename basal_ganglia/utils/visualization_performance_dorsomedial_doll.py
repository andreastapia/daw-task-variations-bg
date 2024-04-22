import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

base_dir = 'basal_ganglia/results/daw_task_dorsomedial/'
base_dir = 'basal_ganglia/results/doll_task_dorsomedial/'

results_folder = os.listdir(base_dir)
simulations_results = []
for idx, simulation in enumerate(results_folder):
    #stim_to_action={0:"(F1,B1)",1:"(F1,B2)",2:"(F2,S1)",3:"(F2,S2)",4:"(T1,B1)",5:"(T1,B2)",6:"(T2,S1)",7:"(T2,S2)"}
    #transitions: 1-> rewarded, 0->not rewarded

    achieved_desired_sequence = np.load(base_dir + simulation+'/achieved_desired_sequence.npy')
    realizable_stimulus = np.load(base_dir + simulation+'/realizable_stimulus.npy')

    data_dict = {
        'achieved': achieved_desired_sequence,
        'realizable': realizable_stimulus
    }

    data = pd.DataFrame(data_dict)
    data.insert(0, 'subject', idx)
    data['trial'] = data.index + 1

    simulations_results.append(data)

all_data = pd.concat(simulations_results, axis=0).reset_index(drop=True)

realizable_data = all_data[all_data['realizable'] == True]
print(realizable_data)

filtered = realizable_data[['subject', 'trial','achieved']].copy()

a_1 = filtered.groupby(['trial','achieved']).count().unstack(fill_value=0).stack()

print(a_1)
# print(a_1.index.unique(level='trial').shape[0])
# print(a_1.loc[[1]].iloc[0]['subject'])
# print(filtered)

#sequence 1 or 5
prop = []
for tr in range(1, a_1.index.unique(level='trial').shape[0]+1):
    trial_prop = []
    total = a_1.loc[[tr]].sum(axis=0)
    for ach in range(a_1.index.unique(level='achieved').shape[0]):
        trial_prop.append(a_1.loc[[tr]].iloc[ach]['subject']/total)
    prop.append(trial_prop)

prop = np.array(prop)
fig, (ax1) = plt.subplots(1)
plt.rcParams["figure.figsize"] = (4,4)
ax1.plot(np.arange(150), prop[:150,1], color='blue')
fig.tight_layout()
#ax1.set_title('Performance over trials')
ax1.set(xlabel='ensayo', ylabel='rendimiento')
ax1.legend(loc='upper right')
plt.show()
