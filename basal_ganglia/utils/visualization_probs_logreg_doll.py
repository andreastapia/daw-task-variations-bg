import os
import numpy as np
import pandas as pd
from scipy.special import expit
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
import statsmodels.api as sm

def plot_probs(probs, legend=True):
    """Plot probabilities."""
    x = (0, 2)
    plt.figure(figsize=(4,4)) 
    plt.bar(x, [probs[i] for i in x], align='center',
            color='tab:orange', label='recompensa')
    x = (1, 3)
    plt.bar(x, [probs[i] for i in x], align='center',
            color='tab:green', label='sin recompensa')
    plt.xticks((0.5, 2.5), ('misma', 'diferente'))
    plt.ylabel('probabilidad de mantener')
    if legend:
        plt.legend(loc='upper right', fontsize='medium')
    plt.ylim(0.0, 1)
    plt.xlim(-0.5, 3.5)
    plt.show()

def get_predictors_pandas(filtered_data):
    logistic_data = filtered_data[['subject','prev_reward','same_start_state','prev_choice']].copy()
    logistic_data.insert(1, 'default_value', 1)
    logistic_data = logistic_data.fillna(0.0)
    logistic_data['prev_reward'] = logistic_data['prev_reward'].apply(encode)
    logistic_data['prev_choice'] = logistic_data['prev_choice'].apply(encode)
    logistic_data['same_rew_inter'] = logistic_data['same_start_state'] * logistic_data['prev_reward']
    logistic_data['same_choice_inter'] = logistic_data['same_start_state'] * logistic_data['prev_choice']
    logistic_data['choice'] = filtered_data['choice'].apply(encode)
    #print(logistic_data)
    
    #print(logistic_data)
    return logistic_data.iloc[:,1:7].values.tolist(), logistic_data.iloc[:,-1].values.tolist()

#0: body parts, 1: scenes
def get_choice(sequence):
    if sequence == 0 or sequence == 1 or sequence == 4 or sequence == 5:
        return 0
    elif sequence == 2 or sequence == 3 or sequence == 6 or sequence == 7:
        return 1

def get_start_state(sequence):
    if sequence == 0 or sequence == 1 or sequence == 2 or sequence == 3:
        return 1
    elif sequence == 4 or sequence == 5 or sequence == 6 or sequence == 7:
        return 2

def encode(value):
    if value == 0.0:
        return value
    elif value == 1.0:
        return value


base_dir = 'basal_ganglia/results/doll_task/'
base_dir = 'basal_ganglia/results/doll_task_ventral/'

results_folder = os.listdir(base_dir)
x, y = [],[]
for idx, simulation in enumerate(results_folder):
    acc_reward = np.load(base_dir + simulation+'/acc_reward.npy')
    #{0:"(1,A)",1:"(1,B)",2:"(1,C)",3:"(1,D)",4:"(2,A)",5:"(2,B)",6:"(2,C)",7:"(2,D)"}
    #transitions: 1-> rewarded, 0->not rewarded
    achieved_sequences = np.load(base_dir + simulation+'/achieved_sequences.npy')
    realizable_stimulus = np.load(base_dir + simulation+'/realizable_stimulus.npy')
    rewarded = np.load(base_dir + simulation+'/rewarded.npy')
    selected_objectives = np.load(base_dir + simulation+'/selected_objectives.npy')

    data_dict = {
        'achieved_sequences': achieved_sequences,
        #'realizable_stimulus': realizable_stimulus,
        'rewarded': rewarded,
        #'selected_objectives': selected_objectives
    }



    data = pd.DataFrame(data_dict)

    data['prev_reward'] = data['rewarded'].shift(1)
    data['choice'] = data['achieved_sequences'].apply(get_choice)
    data['prev_choice'] = data['choice'].shift(1)
    data['start_state'] = data['achieved_sequences'].apply(get_start_state)
    data['prev_start_state'] = data['start_state'].shift(1)
    data['same_start_state'] = np.where( data['prev_start_state'] == data['start_state'], 1, 0)
    data.insert(0, 'subject', idx)
    data = data.fillna(0.0)

    #print(data.iloc[:,:])

    
    #filtered_data = data.iloc[:,:]
    #for the full model with dorsomedial pre-train, adjust
    filtered_data = data.iloc[100:,:]

    xx, yy = get_predictors_pandas(filtered_data)

    M = [0]*len(results_folder)

    M[idx] = 1
    for l in xx:
        x.append(np.hstack(M + l))
    y += yy

logreg = LogisticRegression(fit_intercept=False, C=1e6)
logreg.fit(x, y)


coefs = logreg.coef_[0][
    len(results_folder):(len(results_folder) + 8)]


model = sm.Logit(y, x).fit()
std_errors = model.bse
#print(model.pvalues[-6:])
print(model.summary())
print(coefs)


#prev rewarded, same first state, 
probs = (
    expit(coefs[0] + coefs[1] + coefs[2] + coefs[4]), #rewarded, same
    expit(coefs[0] - coefs[1] + coefs[2] - coefs[4]), #unrewarded, same
    expit(coefs[0] + coefs[1] - coefs[2] - coefs[4]), #rewarded, different
    expit(coefs[0] - coefs[1] - coefs[2] + coefs[4]), #unrewarded, different
)

plot_probs(probs)
del logreg