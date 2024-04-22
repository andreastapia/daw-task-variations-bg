import os
import numpy as np
import pandas as pd
from scipy.special import expit
import matplotlib.pyplot as plt
import matplotlib as mpl
from sklearn.linear_model import LogisticRegression
import statsmodels.api as sm

def plot_probs(probs, legend=True):
    """Plot probabilities."""
    x = (0, 2)
    plt.figure(figsize=(4,4)) 
    plt.bar(x, [probs[i] for i in x], align='center',
            color='tab:orange', label='común')
    x = (1, 3)
    plt.bar(x, [probs[i] for i in x], align='center',
            color='tab:green', label='rara')
    plt.xticks((0.5, 2.5), ('recompensa', 'sin recompensa'))
    plt.ylabel('probabilidad de mantener')
    if legend:
        plt.legend(loc='upper right', fontsize='medium')
    plt.ylim(0.5, 1)
    plt.xlim(-0.5, 3.5)
    
    plt.show()

def get_predictors_pandas(filtered_data):
    logistic_data = filtered_data[['subject','prev_reward', 'prev_transition']]
    logistic_data.insert(1, 'default_value', 1)
    logistic_data = logistic_data.fillna(0.0)
    logistic_data['prev_reward'] = logistic_data['prev_reward'].apply(encode)
    logistic_data['prev_transition'] = logistic_data['prev_transition'].apply(encode)
    logistic_data['prev_rew_tran_inter'] = logistic_data['prev_reward'] * logistic_data['prev_transition']
    logistic_data['same_first_choice'] = np.where((filtered_data['first_action'] == filtered_data['prev_first_action']), 1, 0)

    return logistic_data.iloc[:,1:5].values.tolist(), logistic_data.iloc[:,-1].values.tolist()

def get_first_action(sequence):
    if sequence < 4:
        return '1'
    elif sequence >=4:
        return '2'
    else:
        return None

def encode(value):
    if value == 0.0:
        return value
    elif value == 1.0:
        return value

# base_dir = 'basal_ganglia/results/daw_task/'
base_dir = 'basal_ganglia/results/daw_task_bigger_pretrain/'
base_dir = 'basal_ganglia/results/dezfoulli_task_bigger_pretrain_2/'
base_dir = 'basal_ganglia/results/dezfoulli_task/'
base_dir = 'basal_ganglia/results/daw_task_ventral/'
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
    data['prev_sequence'] = data['achieved_sequences'].shift(1)
    data['prev_transition'] = data['common_transitions'].shift(1)
    data['prev_reward'] = data['rewarded'].shift(1)
    data['prev_first_action'] = data['achieved_sequences'].apply(get_first_action).shift(1)
    data['first_action'] = data['achieved_sequences'].apply(get_first_action)
    data.insert(0, 'subject', idx)
    data = data.fillna(0.0)

    filtered_data = data.iloc[:,:]
    #filtered_data = data.iloc[150:350,:]

    xx, yy = get_predictors_pandas(filtered_data)

    M = [0]*len(results_folder)

    M[idx] = 1
    for l in xx:
        x.append(np.hstack(M + l))
    y += yy

logreg = LogisticRegression(fit_intercept=False, C=1e6)
logreg.fit(x, y)

coefs = logreg.coef_[0][
    len(results_folder):(len(results_folder) + 4)]

model = sm.Logit(y, x).fit()
std_errors = model.bse
#print(model.pvalues[-6:])
print(model.summary())
print(coefs)


probs = (
    expit(coefs[0] + coefs[1] + coefs[2] + coefs[3]), #rewarded, common
    expit(coefs[0] + coefs[1] - coefs[2] - coefs[3]), #rewarded, rare
    expit(coefs[0] - coefs[1] + coefs[2] - coefs[3]), #unrewarded, common
    expit(coefs[0] - coefs[1] - coefs[2] + coefs[3]), #unrewarded, rare
)

print(probs)
plot_probs(probs)
del logreg