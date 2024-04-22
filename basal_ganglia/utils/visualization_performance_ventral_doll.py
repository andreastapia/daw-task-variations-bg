import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

def get_first_action(sequence):
    if sequence == 0 or sequence == 1:
        return 'F1'
    elif sequence == 2 or sequence == 3:
        return 'F2'
    elif sequence == 4 or sequence == 5:
        return 'T1'
    elif sequence == 6 or sequence == 7:
        return 'T2'
    else:
        return None

def get_second_action(sequence):
    if sequence == 0 or sequence == 4:
        return 'B1'
    elif sequence == 1 or sequence == 5:
        return 'B2'
    elif sequence == 2 or sequence == 6:
        return 'S1'
    elif sequence == 3 or sequence == 7:
        return 'S2'
    else:
        return None

base_dir = 'basal_ganglia/results/doll_task_ventral_fixed/'

results_folder = os.listdir(base_dir)
simulations_results = []
for idx, simulation in enumerate(results_folder):
    #stim_to_action={0:"(F1,B1)",1:"(F1,B2)",2:"(F2,S1)",3:"(F2,S2)",4:"(T1,B1)",5:"(T1,B2)",6:"(T2,S1)",7:"(T2,S2)"}
    #transitions: 1-> rewarded, 0->not rewarded

    selected_objectives = np.load(base_dir + simulation+'/selected_objectives.npy')

    data_dict = {
        'selected_objectives': selected_objectives
    }

    data = pd.DataFrame(data_dict)
    data['first_action'] = data['selected_objectives'].apply(get_first_action)
    data['second_action'] = data['selected_objectives'].apply(get_second_action)
    data.insert(0, 'subject', idx)
    data['trial'] = data.index + 1
    data['sequence'] = "(" + data['first_action'] + "," + data['second_action'] + ")"

    simulations_results.append(data)

all_data = pd.concat(simulations_results, axis=0).reset_index(drop=True)

filtered = all_data[['subject', 'trial','sequence']].copy()


#print(all_data)
a_1 = filtered.groupby(['trial','sequence']).count().unstack(fill_value=0).stack()


print(a_1.index.unique(level='trial').shape[0])
print(a_1.loc[[1]].iloc[0]['subject'])
print(filtered)

#sequence 1 or 5
prop = []
for tr in range(1, a_1.index.unique(level='trial').shape[0]+1):
    trial_prop = []
    for seq in range(a_1.index.unique(level='sequence').shape[0]):
        trial_prop.append(a_1.loc[[tr]].iloc[seq]['subject']/len(results_folder))
    prop.append(trial_prop)

plt.rcParams["figure.figsize"] = (4,4)
prop = np.array(prop)
fig, (ax1) = plt.subplots(1)
ax1.plot(np.arange(prop.shape[0]), prop[:,1], label='(F1,B2)', color='blue')
ax1.plot(np.arange(prop.shape[0]), prop[:,5], label='(T1,B2)', color='orange')
ax1.plot(np.arange(prop.shape[0]), prop[:,1] + prop[:,5], label='Opción B2', color='green')
fig.tight_layout()
ax1.set(xlabel='ensayo', ylabel='rendimiento')
ax1.legend(loc='upper right')
plt.show()
