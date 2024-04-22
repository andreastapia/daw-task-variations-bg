import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

base_dir = 'basal_ganglia/results/doll_task_dorsomedial/'
#base_dir = 'basal_ganglia/results/daw_task_dorsomedial/'

simulation_pick =95
trial_pick = 150
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


mpfc_act_np = np.array(mpfc_activities)
gpi_act_np = np.array(gpi_activities)
dfpc_act_np = np.array(dPFC_activities)
objective_learning_act_np = np.array(objective_learning)
gpi_learning_act_np = np.array(gpi_learning)
dpfc_learning_act_np = np.array(dpfc_learning)
strd1_learning_act_np = np.array(strd1_learning)
dopamine_activities_act_np = np.array(dopamine_activities)
strd1_activities_act_np = np.array(strd1_activities)

# print("objective_learning", objective_learning_act_np.shape)
# print("gpi_learning", gpi_learning_act_np.shape)
# print("dpfc_learning", dpfc_learning_act_np.shape)
# print("strd1_learning", strd1_learning_act_np.shape)

# strd1_trial = strd1_activities_act_np[trial_pick].reshape(strd1_activities_act_np.shape[1], 6,8)
# strd1_avg = np.average(strd1_trial, axis=0)
# strd1_avg2 = np.average(strd1_avg, axis=0)
# print("strd1_learning trial", strd1_learning_trial.shape)
# print("strd1_learning avg", strd1_learning_avg.shape)
# print("strd1_learning avg", strd1_learning_avg2.shape)
# print("strd1 avg", strd1_avg2)


strd1_learning_trial = strd1_learning_act_np[trial_pick].reshape(strd1_learning_act_np.shape[1], 6,8)
strd1_learning_avg = np.average(strd1_learning_trial, axis=0)
strd1_learning_avg2 = np.average(strd1_learning_avg, axis=0)
# print("strd1_learning trial", strd1_learning_trial.shape)
# print("strd1_learning avg", strd1_learning_avg.shape)
# print("strd1_learning avg", strd1_learning_avg2.shape)
print("strd1_learning ", strd1_learning_avg2)
print("strd1_learning avg", np.average(strd1_learning_avg2))

# print("dopa_act", dopamine_activities.shape)
# print("dopa_act", dopamine_activities[trial_pick])
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


fig, (ax1, ax2, ax3) = plt.subplots(3)

labels = {
    0:'(F1,B1)',
    1:'(F1,B2)',
    2:'(F2,S1)',
    3:'(F2,S2)',
    4:'(T1,B1)',
    5:'(T1,B2)',
    6:'(T2,S1)',
    7:'(T2,S2)'
}

for i, color in enumerate(['dimgray','indianred','darkred','tan','slateblue','navy','fuchsia','pink']):
    ax1.plot(np.arange(mpfc_act_np.shape[1]), mpfc_act_np[trial_pick][:,i], label=labels[i], color=color)

ax1.set_title('Actividad del mPFC a lo largo del tiempo (ensayo {trial_pick})'.format(trial_pick=trial_pick))

ax2.plot(np.arange(gpi_act_np.shape[1]), gpi_act_np[trial_pick][:,0], label='(F1)')
ax2.plot(np.arange(gpi_act_np.shape[1]), gpi_act_np[trial_pick][:,1], label='(F2)')
ax2.plot(np.arange(gpi_act_np.shape[1]), gpi_act_np[trial_pick][:,2], label='(T1)')
ax2.plot(np.arange(gpi_act_np.shape[1]), gpi_act_np[trial_pick][:,3], label='(T2)')
ax2.plot(np.arange(gpi_act_np.shape[1]), gpi_act_np[trial_pick][:,4], label='(B1)')
ax2.plot(np.arange(gpi_act_np.shape[1]), gpi_act_np[trial_pick][:,5], label='(B2)')
ax2.plot(np.arange(gpi_act_np.shape[1]), gpi_act_np[trial_pick][:,6], label='(S1)')
ax2.plot(np.arange(gpi_act_np.shape[1]), gpi_act_np[trial_pick][:,7], label='(S2)')
ax2.set_title('Actividad del GPi a lo largo del tiempo (ensayo {trial_pick})'.format(trial_pick=trial_pick))

ax3.plot(np.arange(dfpc_act_np.shape[1]), dfpc_act_np[trial_pick][:,0], label='(F1)')
ax3.plot(np.arange(dfpc_act_np.shape[1]), dfpc_act_np[trial_pick][:,1], label='(F2)')
ax3.plot(np.arange(dfpc_act_np.shape[1]), dfpc_act_np[trial_pick][:,2], label='(T1)')
ax3.plot(np.arange(dfpc_act_np.shape[1]), dfpc_act_np[trial_pick][:,3], label='(T2)')
ax3.plot(np.arange(dfpc_act_np.shape[1]), dfpc_act_np[trial_pick][:,4], label='(B1)')
ax3.plot(np.arange(dfpc_act_np.shape[1]), dfpc_act_np[trial_pick][:,5], label='(B2)')
ax3.plot(np.arange(dfpc_act_np.shape[1]), dfpc_act_np[trial_pick][:,6], label='(S1)')
ax3.plot(np.arange(dfpc_act_np.shape[1]), dfpc_act_np[trial_pick][:,7], label='(S2)')
ax3.set_title('Actividad del dPFC a lo largo del tiempo (ensayo {trial_pick})'.format(trial_pick=trial_pick))

dopa_avg = np.average(dopamine_activities, axis=1)
print(dopa_avg.shape)


# ax4.plot(np.arange(dopa_avg.shape[0]), dopa_avg, label='(F1,B1)')
# ax4.set_title('Average dopamine (SNc) over trials')

ax1.legend(loc='upper right', title='Secuencia')
ax3.legend(loc='upper right', title='Opcion')

plt.show()