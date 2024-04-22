import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.special import expit
import statsmodels.formula.api as smf

def plot_probs(probs, legend=True):
    """Plot probabilities."""
    x = (0, 2)
    plt.bar(x, [probs[i] for i in x], align='center',
            color='tab:orange', label='common')
    x = (1, 3)
    plt.bar(x, [probs[i] for i in x], align='center',
            color='tab:green', label='rare')
    plt.xticks((0.5, 2.5), ('rewarded', 'unrewarded'))
    plt.ylabel('stay probability')
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
    logistic_data['same_first_choice'] = np.where((filtered_data['first_action'] == filtered_data['prev_first_action']), 1, -1)

    return logistic_data

def get_first_action(sequence):
    if sequence < 4:
        return 'A'
    elif sequence >=4:
        return 'B'
    else:
        return None

def encode(value):
    if value == 0.0:
        return -1.0
    elif value == 1.0:
        return value

#base_dir = 'basal_ganglia/results/daw_task/'
#base_dir = 'basal_ganglia/results/doll_task/'
base_dir = 'basal_ganglia/results/dezfoulli_task/'

#TODO: replicate second stage choices graphs

results_folder = os.listdir(base_dir)
simulations_results = []
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
        'realizable_stimulus': realizable_stimulus,
        'rewarded': rewarded,
        'selected_objectives': selected_objectives
    }

    data = pd.DataFrame(data_dict)
    data['prev_sequence'] = data['achieved_sequences'].shift(1)
    data['prev_transition'] = data['common_transitions'].shift(1)
    data['prev_reward'] = data['rewarded'].shift(1)
    data['prev_first_action'] = data['achieved_sequences'].apply(get_first_action).shift(1)
    data['first_action'] = data['achieved_sequences'].apply(get_first_action)
    data.insert(0, 'subject', idx)

    filtered_data = data.iloc[100:,:]

    simulations_results.append(filtered_data)

all_data = pd.concat(simulations_results, axis=0).reset_index(drop=True)
reg_input = get_predictors_pandas(all_data)

#print(all_data[all_data['subject'] == 0])

print(all_data)
print(reg_input)

formula = "same_first_choice ~ prev_reward + prev_transition + prev_rew_tran_inter"

md = smf.mixedlm(formula, reg_input, groups=reg_input['subject'])
output = md.fit()
print(output.summary())

del reg_input

axes = plt.subplot(1, 2, 0 + 1)
axes.spines['right'].set_color('none')
axes.spines['top'].set_color('none')
axes.xaxis.set_ticks_position('bottom')
axes.yaxis.set_ticks_position('left')
plt.title('Basal Ganglia Model')

coefs = output.params
print("coefficients\n", coefs)
print("p-values\n", output.pvalues)
#first value is the constant value (1)
#second value is prev reward
#third value is prev transition
#fourth value is the interaction between
#first expit (rewarded, common, +interaction)
#second expit (rewarded, rare, -interaction)
#third expit (unrewarded, common, -interaction)
#fourth expit (unrewarded, rare, +interaction)
#inverse logit for each case
probs = (
    expit(coefs[0] + coefs[1] + coefs[2] + coefs[3]),
    expit(coefs[0] + coefs[1] - coefs[2] - coefs[3]),
    expit(coefs[0] - coefs[1] + coefs[2] - coefs[3]),
    expit(coefs[0] - coefs[1] - coefs[2] + coefs[3]),
)

plot_probs(probs)
del simulations_results
del all_data
