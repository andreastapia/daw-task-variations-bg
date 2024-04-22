import os
import numpy as np
import pandas as pd
from scipy.special import expit
import matplotlib.pyplot as plt
import matplotlib as mpl
from sklearn.linear_model import LogisticRegression

def plot_probs(probs, legend=True):
    """Plot probabilities."""
    x = (0, 2)
    plt.figure(figsize=(4,4)) 
    plt.bar(x, [probs[i] for i in x], align='center',
            color='tab:orange', label='misma')
    x = (1, 3)
    plt.bar(x, [probs[i] for i in x], align='center',
            color='tab:green', label='diferente')
    plt.xticks((0.5, 2.5), ('recompensa', 'sin recompensa'))
    plt.ylabel('probabilidad de mantener')
    if legend:
        plt.legend(loc='upper right', fontsize='medium')
    plt.ylim(0.0, 1.0)
    plt.xlim(-0.5, 3.5)
    
    plt.show()

def get_predictors_pandas(filtered_data):
    logistic_data = filtered_data[['subject','rewarded']]
    logistic_data.insert(1, 'default_value', 1)
    logistic_data = logistic_data.fillna(0.0)
    logistic_data['same_second_state'] = np.where(filtered_data['second_state'] == filtered_data['next_second_state'],1,0)
    logistic_data['rew_sec_state_inter'] = logistic_data['rewarded'] * logistic_data['same_second_state']
    logistic_data['same_second_choice'] = np.where((filtered_data['second_action'] == filtered_data['next_second_action']), 1, 0)
    logistic_data['same_first_action'] = np.where((filtered_data['first_action'] == filtered_data['next_first_action']), 1, 0)
    logistic_data = logistic_data[logistic_data['same_first_action'] == 1]
    return logistic_data.iloc[:,1:5].values.tolist(), logistic_data.iloc[:,-2].values.tolist()

def get_first_action(sequence):
    if sequence <= 3:
        return 'A1'
    elif sequence >= 4:
        return 'A2'
    else:
        return None

def get_second_action(sequence):
    if sequence == 0 or sequence == 2 or sequence == 4 or sequence == 6:
        return 'A1'
    elif sequence == 1 or sequence == 3 or sequence == 5 or sequence == 7:
        return 'A2'
    else:
        return None

def get_second_state(sequence):
    if sequence == 0 or sequence == 1 or sequence == 4 or sequence == 5:
        return 'S1'
    elif sequence == 2 or sequence == 3 or sequence == 6 or sequence == 7:
        return 'S2'
    else:
        return None

def encode(value):
    if value == 0.0:
        return value
    elif value == 1.0:
        return value

base_dir = 'basal_ganglia/results/dezfoulli_task/'
base_dir = 'basal_ganglia/results/dezfoulli_task_ventral/'

results_folder = os.listdir(base_dir)
x, y = [],[]
for idx, simulation in enumerate(results_folder):
    acc_reward = np.load(base_dir + simulation+'/acc_reward.npy')
    #{0:"(1,A)",1:"(1,B)",2:"(1,C)",3:"(1,D)",4:"(2,A)",5:"(2,B)",6:"(2,C)",7:"(2,D)"}
    #transitions: 1-> rewarded, 0->not rewarded
    achieved_sequences = np.load(base_dir + simulation+'/achieved_sequences.npy')
    common_transitions = np.load(base_dir + simulation+'/common_transitions.npy')
    realizable_stimulus = np.load(base_dir + simulation+'/realizable_stimulus.npy')
    rewarded = np.load(base_dir + simulation+'/rewarded.npy')
    selected_objectives = np.load(base_dir + simulation+'/selected_objectives.npy')

    data_dict = {
        'achieved_sequences': achieved_sequences,
        'common_transitions': common_transitions,
        #'realizable_stimulus': realizable_stimulus,
        'rewarded': rewarded,
        #'selected_objectives': selected_objectives
    }

    data = pd.DataFrame(data_dict)
    data['second_action'] = data['achieved_sequences'].apply(get_second_action)
    data['next_second_action'] = data['second_action'].shift(-1)
    data['first_action'] = data['achieved_sequences'].apply(get_first_action)
    data['next_first_action'] = data['first_action'].shift(-1)
    data['second_state'] = data['achieved_sequences'].apply(get_second_state)
    data['next_second_state'] = data['second_state'].shift(-1)
    data.insert(0, 'subject', idx)
    data = data.fillna(0.0)
    #print(data)
    #filtered_data = data[data['second_state'] != data['next_second_state']]

    
    #filtered_data = data.iloc[:,:]
    filtered_data = data.iloc[:100,:]
    print(filtered_data)

    xx, yy = get_predictors_pandas(filtered_data)

    M = [0]*len(results_folder)

    M[idx] = 1
    for l in xx:
        x.append(np.hstack(M + l))
    y += yy

logreg = LogisticRegression(fit_intercept=False, C=1e6)
logreg.fit(x, y)
del x
del y

coefs = logreg.coef_[0][
    len(results_folder):(len(results_folder) + 4)]
probs = (
    expit(coefs[0] + coefs[1] + coefs[2] + coefs[3]), #rewarded, same second state
    expit(coefs[0] + coefs[1] - coefs[2] - coefs[3]), #rewarded, diff second state
    expit(coefs[0] - coefs[1] + coefs[2] - coefs[3]), #unrewarded, same second state
    expit(coefs[0] - coefs[1] - coefs[2] + coefs[3]), #unrewarded, diff second state
)

plot_probs(probs)
del logreg