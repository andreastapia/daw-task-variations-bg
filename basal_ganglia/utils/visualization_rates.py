import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

base_dir = 'basal_ganglia/results/daw_task_ventral_fixed/'
base_dir = 'basal_ganglia/results/daw_task/'
base_dir = 'basal_ganglia/results/daw_task_ventral/'
simulation_pick = 122
trial_pick = 210
results_folder = os.listdir(base_dir)
results_folder.sort()
simulations_results = []
for idx, simulation in enumerate(results_folder):
    if idx == simulation_pick:    
        mpfc_activities = np.load(base_dir + simulation+'/mpfc_activities.npy')
        vp_activities = np.load(base_dir + simulation+'/vp_activities.npy')
        vp_ind_activities = np.load(base_dir + simulation+'/vp_ind_activities.npy')
        hippo_activities = np.load(base_dir + simulation+'/hippo_activities.npy')
        naccd1_activities = np.load(base_dir + simulation+'/naccd1_activities.npy')
        naccd2_activities = np.load(base_dir + simulation+'/naccd2_activities.npy')
        thal_activities = np.load(base_dir + simulation+'/thal_activities.npy')
        selected_objectives = np.load(base_dir + simulation+'/selected_objectives.npy')
        rewards_over_trial = np.load(base_dir + simulation+'/rewards_over_time.npy')


mpfc_act_np = np.array(mpfc_activities)
vp_act_np = np.array(vp_activities)
vp_ind_act_np = np.array(vp_ind_activities)
hippo_act_np = np.array(hippo_activities)
strd1_act_np = np.array(naccd1_activities)
strd2_act_np = np.array(naccd2_activities)
thal_act_np = np.array(thal_activities)



print("mpfc", mpfc_act_np.shape)
print("mpfc", np.average(mpfc_act_np[trial_pick], axis=0).shape)
print("mpfc", np.average(mpfc_act_np[trial_pick], axis=0))

print("vp_dir", vp_act_np.shape)
print("vp_dir", np.average(vp_act_np[trial_pick], axis=0).shape)
print("vp_dir", np.average(vp_act_np[trial_pick], axis=0))

print("vp_ind", vp_ind_act_np.shape)
print("vp_ind", np.average(vp_ind_act_np[trial_pick], axis=0).shape)
print("vp_ind", np.average(vp_ind_act_np[trial_pick], axis=0))

print("thal", thal_act_np.shape)
print("thal", np.average(thal_act_np[trial_pick], axis=0).shape)
print("thal", np.average(thal_act_np[trial_pick], axis=0))

hippo_trial = hippo_act_np[trial_pick].reshape(hippo_act_np.shape[1], 4,6)
print("hippo", hippo_trial.shape)
hippo_avg = np.average(hippo_trial, axis=1)

strd1_trial = strd1_act_np[trial_pick].reshape(strd1_act_np.shape[1], 4,6)
strd1_avg = np.average(strd1_trial, axis=1)
strd1_avg = np.average(strd1_avg, axis=1)
print("strd1", strd1_trial.shape)
print("strd1", strd1_avg.shape)


strd2_trial = strd2_act_np[trial_pick].reshape(strd2_act_np.shape[1], 4,6)
print("strd2", strd2_trial.shape)
strd2_avg = np.average(strd2_trial, axis=1)

vp_dir_trial = vp_act_np[trial_pick]

vp_indir_trial = vp_ind_act_np[trial_pick]

thal_trial = thal_act_np[trial_pick]

fig, ((ax1, ax2), (ax3, ax4), (ax5, ax6), (ax7, ax8)) = plt.subplots(4, 2)
fig.tight_layout()

sequences = {
    0:'(1,A)',
    1:'(1,B)',
    2:'(1,C)',
    3:'(1,D)',
    4:'(2,A)',
    5:'(2,B)',
    6:'(2,C)',
    7:'(2,D)'
}

options = {
    0:'1',
    1:'2',
    2:'A',
    3:'B',
    4:'C',
    5:'D'
}

for i, color in enumerate(['dimgray','indianred','darkred','tan','slateblue','navy']):
    ax1.plot(np.arange(hippo_avg.shape[0]), hippo_avg[:,i], label=options[i], color=color)
ax1.set_title('Activity of Hippo over time (trial {trial_pick})'.format(trial_pick=trial_pick))
ax1.set(xlabel='time (ms)', ylabel='firing rate')
ax1.legend(loc='upper right', title='Sequence')

ax2.plot(np.arange(strd1_avg.shape[0]), strd1_avg[:])
ax2.set_title('Mean Activity of STR D1 over time (trial {trial_pick})'.format(trial_pick=trial_pick))
ax2.set(xlabel='time (ms)', ylabel='firing rate')

for i, color in enumerate(['dimgray','indianred','darkred','tan','slateblue','navy','fuchsia','pink']):
    ax3.plot(np.arange(vp_dir_trial.shape[0]), vp_dir_trial[:,i], label=sequences[i], color=color)
ax3.set_title('Activity of VP direct over time (trial {trial_pick})'.format(trial_pick=trial_pick))
ax3.set(xlabel='time (ms)', ylabel='firing rate')
ax3.legend(loc='upper right', title='Sequence')

ax4.plot(np.arange(vp_indir_trial.shape[0]), vp_indir_trial[:,0], label='(F1,B1)')
ax4.plot(np.arange(vp_indir_trial.shape[0]), vp_indir_trial[:,1], label='(F1,B2)')
ax4.plot(np.arange(vp_indir_trial.shape[0]), vp_indir_trial[:,2], label='(F2,S1)')
ax4.plot(np.arange(vp_indir_trial.shape[0]), vp_indir_trial[:,3], label='(F2,S2)')
ax4.plot(np.arange(vp_indir_trial.shape[0]), vp_indir_trial[:,4], label='(T1,B1)')
ax4.plot(np.arange(vp_indir_trial.shape[0]), vp_indir_trial[:,5], label='(T1,B2)')
ax4.plot(np.arange(vp_indir_trial.shape[0]), vp_indir_trial[:,6], label='(T2,S1)')
ax4.plot(np.arange(vp_indir_trial.shape[0]), vp_indir_trial[:,7], label='(T2,S2)')
ax4.set_title('Activity of VP indirect over time (trial {trial_pick})'.format(trial_pick=trial_pick))
ax4.set(xlabel='time (ms)', ylabel='firing rate')
ax4.legend(loc='upper right', title='Sequence')


ax5.plot(np.arange(mpfc_act_np.shape[1]), mpfc_act_np[trial_pick][:,0], label='(F1,B1)')
ax5.plot(np.arange(mpfc_act_np.shape[1]), mpfc_act_np[trial_pick][:,1], label='(F1,B2)')
ax5.plot(np.arange(mpfc_act_np.shape[1]), mpfc_act_np[trial_pick][:,2], label='(F2,S1)')
ax5.plot(np.arange(mpfc_act_np.shape[1]), mpfc_act_np[trial_pick][:,3], label='(F2,S2)')
ax5.plot(np.arange(mpfc_act_np.shape[1]), mpfc_act_np[trial_pick][:,4], label='(T1,B1)')
ax5.plot(np.arange(mpfc_act_np.shape[1]), mpfc_act_np[trial_pick][:,5], label='(T1,B2)')
ax5.plot(np.arange(mpfc_act_np.shape[1]), mpfc_act_np[trial_pick][:,6], label='(T2,S1)')
ax5.plot(np.arange(mpfc_act_np.shape[1]), mpfc_act_np[trial_pick][:,7], label='(T2,S2)')

ax5.set_title('Activity of mPFC over time (trial {trial_pick})'.format(trial_pick=trial_pick))
ax5.set(xlabel='time (ms)', ylabel='firing rate')
ax5.legend(loc='upper right', title='Sequence')

vp_dir_trial_avg = np.empty((vp_act_np.shape[0], vp_act_np.shape[2]))
for idx, act in enumerate(vp_act_np):    
    vp_dir_trial_avg[idx] =  np.average(act, axis=0)

ax6.plot(np.arange(vp_dir_trial_avg.shape[0]), vp_dir_trial_avg[:,0], label='(F1,B1)')
ax6.plot(np.arange(vp_dir_trial_avg.shape[0]), vp_dir_trial_avg[:,1], label='(F1,B2)')
ax6.plot(np.arange(vp_dir_trial_avg.shape[0]), vp_dir_trial_avg[:,2], label='(F2,S1)')
ax6.plot(np.arange(vp_dir_trial_avg.shape[0]), vp_dir_trial_avg[:,3], label='(F2,S2)')
ax6.plot(np.arange(vp_dir_trial_avg.shape[0]), vp_dir_trial_avg[:,4], label='(T1,B1)')
ax6.plot(np.arange(vp_dir_trial_avg.shape[0]), vp_dir_trial_avg[:,5], label='(T1,B2)')
ax6.plot(np.arange(vp_dir_trial_avg.shape[0]), vp_dir_trial_avg[:,6], label='(T2,S1)')
ax6.plot(np.arange(vp_dir_trial_avg.shape[0]), vp_dir_trial_avg[:,7], label='(T2,S2)')

ax6.set_title('Mean activity of VP over trials')
ax6.set(xlabel='trial', ylabel='neuron activity')
ax6.legend(loc='upper right', title='Sequence')

# vp_ind_trial_avg = np.empty((vp_ind_act_np.shape[0], vp_ind_act_np.shape[2]))
# for idx, act in enumerate(vp_ind_act_np):    
#     vp_ind_trial_avg[idx] =  np.average(act, axis=0)
# ax7.plot(np.arange(vp_ind_trial_avg.shape[0]), vp_ind_trial_avg[:,0], label='(F1,B1)')
# ax7.plot(np.arange(vp_ind_trial_avg.shape[0]), vp_ind_trial_avg[:,1], label='(F1,B2)')
# ax7.plot(np.arange(vp_ind_trial_avg.shape[0]), vp_ind_trial_avg[:,2], label='(F2,S1)')
# ax7.plot(np.arange(vp_ind_trial_avg.shape[0]), vp_ind_trial_avg[:,3], label='(F2,S2)')
# ax7.plot(np.arange(vp_ind_trial_avg.shape[0]), vp_ind_trial_avg[:,4], label='(T1,B1)')
# ax7.plot(np.arange(vp_ind_trial_avg.shape[0]), vp_ind_trial_avg[:,5], label='(T1,B2)')
# ax7.plot(np.arange(vp_ind_trial_avg.shape[0]), vp_ind_trial_avg[:,6], label='(T2,S1)')
# ax7.plot(np.arange(vp_ind_trial_avg.shape[0]), vp_ind_trial_avg[:,7], label='(T2,S2)')

# ax7.set_title('Mean activity of VP ind over trials')
# ax7.set(xlabel='trial', ylabel='neuron activity')
# ax7.legend(loc='upper right', title='Sequence')

ax7.plot(np.arange(thal_trial.shape[0]), thal_trial[:,0], label='(F1,B1)')
ax7.plot(np.arange(thal_trial.shape[0]), thal_trial[:,1], label='(F1,B2)')
ax7.plot(np.arange(thal_trial.shape[0]), thal_trial[:,2], label='(F2,S1)')
ax7.plot(np.arange(thal_trial.shape[0]), thal_trial[:,3], label='(F2,S2)')
ax7.plot(np.arange(thal_trial.shape[0]), thal_trial[:,4], label='(T1,B1)')
ax7.plot(np.arange(thal_trial.shape[0]), thal_trial[:,5], label='(T1,B2)')

ax7.set_title('Activity of Thal over time (trial {trial_pick})'.format(trial_pick=trial_pick))
ax7.set(xlabel='time (ms)', ylabel='firing rate')
ax7.legend(loc='upper right', title='Sequence')

ax8.plot(np.arange(rewards_over_trial.shape[0]), rewards_over_trial[:,0], label='B1')
ax8.plot(np.arange(rewards_over_trial.shape[0]), rewards_over_trial[:,1], label='B2')
ax8.plot(np.arange(rewards_over_trial.shape[0]), rewards_over_trial[:,2], label='S1')
ax8.plot(np.arange(rewards_over_trial.shape[0]), rewards_over_trial[:,3], label='S2')

ax8.set_ylim([0.0, 1.0])
ax8.set_title('Reward probability of second state option over time')
ax8.set(xlabel='trial', ylabel='probability')
ax8.legend(loc='upper right', title='Option')
plt.show()