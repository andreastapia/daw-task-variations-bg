import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

def get_first_action(sequence):
    if sequence < 4:
        return '1'
    elif sequence >=4:
        return '2'
    else:
        return None

def get_second_action(sequence):
    if sequence == 0 or sequence == 4:
        return 'A'
    elif sequence == 1 or sequence == 5:
        return 'B'
    elif sequence == 2 or sequence == 6:
        return 'C'
    elif sequence == 3 or sequence == 7:
        return 'D'
    else:
        return None

base_dir = 'basal_ganglia/results/daw_task_ventral_fixed/'

results_folder = os.listdir(base_dir)
simulations_results = []
for idx, simulation in enumerate(results_folder):
    #{0:"(1,A)",1:"(1,B)",2:"(1,C)",3:"(1,D)",4:"(2,A)",5:"(2,B)",6:"(2,C)",7:"(2,D)"}
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


a_1 = filtered.groupby(['trial','sequence']).count().unstack(fill_value=0).stack()


print(a_1.index.unique(level='trial').shape[0])
print(a_1.loc[[1]].iloc[0]['subject'])
print(filtered)

prop = []
for tr in range(1, a_1.index.unique(level='trial').shape[0]+1):
    trial_prop = []
    for seq in range(a_1.index.unique(level='sequence').shape[0]):
        trial_prop.append(a_1.loc[[tr]].iloc[seq]['subject']/len(results_folder))
    prop.append(trial_prop)

prop = np.array(prop)
fig, (ax1) = plt.subplots(1)
ax1.plot(np.arange(prop.shape[0]), prop[:,2], label='(1,C)', color='blue')
ax1.plot(np.arange(prop.shape[0]), prop[:,6], label='(2,C)', color='orange')
ax1.plot(np.arange(prop.shape[0]), prop[:,2] + prop[:,6], label='Option A', color='green')
fig.tight_layout()
ax1.set_title('Performance over trials')
ax1.set(xlabel='trial', ylabel='performance')
ax1.legend(loc='upper right')
plt.show()
