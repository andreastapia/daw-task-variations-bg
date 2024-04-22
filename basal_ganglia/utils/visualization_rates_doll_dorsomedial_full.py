import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

base_dir = 'basal_ganglia/results/doll_task_dorsomedial/'

simulation_pick =134
trial_pick = 8
results_folder = os.listdir(base_dir)
results_folder.sort()
simulations_results = []
for idx, simulation in enumerate(results_folder):
    if idx == simulation_pick:    
        mpfc_activities = np.load(base_dir + simulation+'/mpfc_activities.npy')
        gpi_activities = np.load(base_dir + simulation+'/gpi_activities.npy')
        dPFC_activities = np.load(base_dir + simulation+'/dPFC_activities.npy')
        strd1_activities = np.load(base_dir + simulation+'/strd1_activities.npy')
        objective_learning = np.load(base_dir + simulation+'/objective_learning.npy')
        gpi_learning = np.load(base_dir + simulation+'/gpi_learning.npy')
        dpfc_learning = np.load(base_dir + simulation+'/dpfc_learning.npy')
        strd1_learning = np.load(base_dir + simulation+'/strd1_learning.npy')
        achieved_desired_sequence = np.load(base_dir + simulation+'/achieved_desired_sequence.npy')
        dopamine_activities = np.load(base_dir + simulation+'/dopamine_activities.npy')
        feedback_activities = np.load(base_dir + simulation+'/feedback_activities.npy')
        thal_activities = np.load(base_dir + simulation+'/thal_activities.npy')


mpfc_act_np = np.array(mpfc_activities)
gpi_act_np = np.array(gpi_activities)
dfpc_act_np = np.array(dPFC_activities)
objective_learning_act_np = np.array(objective_learning)
gpi_learning_act_np = np.array(gpi_learning)
dpfc_learning_act_np = np.array(dpfc_learning)
strd1_learning_act_np = np.array(strd1_learning)
dopamine_activities_act_np = np.array(dopamine_activities)
strd1_activities_act_np = np.array(strd1_activities)
feedback_act_np = np.array(feedback_activities)
thal_act_np = np.array(thal_activities)

# print("objective_learning", objective_learning_act_np.shape)
# print("gpi_learning", gpi_learning_act_np.shape)
# print("dpfc_learning", dpfc_learning_act_np.shape)
print("strd1_learning", strd1_activities_act_np.shape)
print("strd1_learning", strd1_learning_act_np.shape)

strd1_trial = strd1_activities_act_np[trial_pick].reshape(strd1_activities_act_np.shape[1], 6,8)
strd1_trial_avg = np.average(strd1_trial, axis=1)
# strd1_trial_avg = np.average(strd1_trial, axis=0)
# strd1_trial_avg2 = np.average(strd1_trial_avg, axis=0)
# print("strd1 trial", strd1_trial.shape)
print("strd1 avg", strd1_trial_avg.shape)
#print("strd1 avg avg", strd1_trial_avg2.shape)
# print("strd1 avg", strd1_avg2)


strd1_learning_trial = strd1_learning_act_np[trial_pick].reshape(strd1_learning_act_np.shape[1], 6,8)
strd1_learning_trial_avg = np.average(strd1_learning_trial, axis=1)
# strd1_learning_avg = np.average(strd1_learning_trial, axis=0)
# strd1_learning_avg2 = np.average(strd1_learning_avg, axis=0)
# print("strd1_learning trial", strd1_learning_trial.shape)
# print("strd1_learning avg", strd1_learning_avg.shape)
# print("strd1_learning avg", strd1_learning_avg2.shape)
# print("strd1_learning ", strd1_learning_avg2)
# print("strd1_learning avg", np.average(strd1_learning_avg2))

print("dopa_act", dopamine_activities.shape)
print("dopa_act", dopamine_activities[trial_pick].shape)
# print("dopa_act", np.average(dopamine_activities[trial_pick], axis=0))

# print("gpi", gpi_act_np.shape)
# print("gpi", np.average(gpi_act_np[trial_pick], axis=0).shape)
# print("gpi", np.average(gpi_act_np[trial_pick], axis=0))

# print("strd1", strd1_activities.shape)
# print("strd1", np.average(strd1_activities[trial_pick], axis=0).shape)
# print("strd1", np.average(strd1_activities[trial_pick], axis=0))

# print("gpi learning", gpi_learning_act_np.shape)
# print("gpi learning", np.average(gpi_learning_act_np[trial_pick], axis=0).shape)
print("gpi learning", np.average(gpi_learning_act_np[trial_pick], axis=0))
print("gpi learning avg", np.average(np.average(gpi_learning_act_np[trial_pick], axis=0)))

# print("dpfc learning", dpfc_learning_act_np.shape)
# print("dpfc learning", np.average(dpfc_learning_act_np[trial_pick], axis=0).shape)
print("dpfc learning", np.average(dpfc_learning_act_np[trial_pick], axis=0))

print("objective learning", np.average(objective_learning_act_np[trial_pick], axis=0))

print("feedback", feedback_act_np.shape)
feedback_trial = feedback_act_np[trial_pick]
print("feedback", feedback_act_np.shape)

sequences = {
    0:'(F1,B1)',
    1:'(F1,B2)',
    2:'(F2,S1)',
    3:'(F2,S2)',
    4:'(T1,B1)',
    5:'(T1,B2)',
    6:'(T2,S1)',
    7:'(T2,S2)'
}

options = {
    0:'F1',
    1:'F2',
    2:'T1',
    3:'T2',
    4:'B1',
    5:'B2',
    6:'S1',
    7:'S2'
}

fig, ((ax1, ax2), (ax3, ax4), (ax5, ax6), (ax7, ax8)) = plt.subplots(4, 2)
fig.tight_layout()


for i, color in enumerate(['dimgray','indianred','darkred','tan','slateblue','navy','fuchsia','pink']):
    ax1.plot(np.arange(mpfc_act_np.shape[1]), mpfc_act_np[trial_pick][:,i], label=sequences[i], color=color)

ax1.set_title('Actividad del mPFC a lo largo del tiempo (ensayo {trial_pick})'.format(trial_pick=trial_pick))

for i, color in enumerate(['dimgray','indianred','darkred','tan','slateblue','navy','fuchsia','pink']):
    ax3.plot(np.arange(strd1_trial_avg.shape[0]), strd1_trial_avg[:,i], label=options[i], color=color)
ax3.set_title('StrD1 activities over time')


for i, color in enumerate(['dimgray','indianred','darkred','tan','slateblue','navy','fuchsia','pink']):
    ax2.plot(np.arange(dfpc_act_np.shape[1]), dfpc_act_np[trial_pick][:,i], label=options[i])
ax2.set_title('Actividad del dPFC a lo largo del tiempo (ensayo {trial_pick})'.format(trial_pick=trial_pick))

for i, color in enumerate(['dimgray','indianred','darkred','tan','slateblue','navy','fuchsia','pink']):
    ax4.plot(np.arange(gpi_act_np.shape[1]), gpi_act_np[trial_pick][:,i], label=options[i])
ax4.set_title('Actividad del GPi a lo largo del tiempo (ensayo {trial_pick})'.format(trial_pick=trial_pick))


for i, color in enumerate(['dimgray','indianred','darkred','tan','slateblue','navy','fuchsia','pink']):
    ax7.plot(np.arange(dopamine_activities_act_np.shape[0]), dopamine_activities_act_np[:,i], label=options[i], color=color)
ax7.set_title('Actividad del SNc a lo largo de los ensayos')

for i, color in enumerate(['dimgray','indianred','darkred','tan','slateblue','navy','fuchsia','pink']):
    ax6.plot(np.arange(thal_act_np.shape[1]), thal_act_np[trial_pick][:,i], label=options[i])
ax6.set_title('Actividad del tálamo a lo largo del tiempo (ensayo {trial_pick})'.format(trial_pick=trial_pick))

# dopa_avg = np.average(dopamine_activities, axis=1)
# ax6.plot(np.arange(dopa_avg.shape[0]), dopa_avg)
# ax6.set_title('Actividad promedio del SNc a lo largo de los ensayos')

for i, color in enumerate(['dimgray','indianred','darkred','tan','slateblue','navy','fuchsia','pink']):
    ax5.plot(np.arange(strd1_learning_trial_avg.shape[0]), strd1_learning_trial_avg[:,i], label=options[i], color=color)
ax5.set_title('Actividad promedio del STR D1 a lo largo del tiempo')

for i, color in enumerate(['dimgray','indianred','darkred','tan','slateblue','navy','fuchsia','pink']):
    ax8.plot(np.arange(feedback_trial.shape[0]), feedback_trial[:,i], label=options[i])
ax8.set_title('Actividad del feedback a lo largo del tiempo')

ax1.legend(loc='upper right', title='Secuencia')
ax2.legend(loc='upper right', title='Opcion')

print(dopamine_activities_act_np)

plt.show()