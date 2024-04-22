import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

#base_dir = 'basal_ganglia/results/dezfoulli_task/'
#base_dir = 'basal_ganglia/results/dezfoulli_task_3000/'
base_dir = 'basal_ganglia/results/dezfoulli_task_ventral/'
#base_dir = 'basal_ganglia/results/daw_task_ventral/'
#base_dir = 'basal_ganglia/results/dezfoulli_task_ventral_3000/'
#base_dir = 'basal_ganglia/results/stored/dezfoulli_with_rates/'
#base_dir = 'basal_ganglia/results/stored/dezfoulli_task_ventral_fixed/'

simulation_pick = 95
trial_pick = 200
results_folder = os.listdir(base_dir)
results_folder.sort()
simulations_results = []
for idx, simulation in enumerate(results_folder):
    if idx == simulation_pick:    
        mpfc_activities = np.load(base_dir + simulation+'/mpfc_activities.npy')
        vp_activities = np.load(base_dir + simulation+'/vp_activities.npy')
        vp_ind_activities = np.load(base_dir + simulation+'/vp_ind_activities.npy')
        selected_objectives = np.load(base_dir + simulation+'/selected_objectives.npy')
        rewards_over_trial = np.load(base_dir + simulation+'/rewards_over_time.npy')


mpfc_act_np = np.array(mpfc_activities)
vp_act_np = np.array(vp_activities)
vp_ind_act_np = np.array(vp_ind_activities)


print("mpfc", mpfc_act_np.shape)
print("mpfc", np.average(mpfc_act_np[trial_pick], axis=0).shape)
print("mpfc", np.average(mpfc_act_np[trial_pick], axis=0))

print("vp_dir", vp_act_np.shape)
print("vp_dir", np.average(vp_act_np[trial_pick], axis=0).shape)
print("vp_dir", np.average(vp_act_np[trial_pick], axis=0))

print("vp_ind", vp_ind_act_np.shape)
print("vp_ind", np.average(vp_ind_act_np[trial_pick], axis=0).shape)
print("vp_ind", np.average(vp_ind_act_np[trial_pick], axis=0))

grid = plt.GridSpec(4, 2)
fig = plt.figure(figsize=(10,15))

ax1 = plt.subplot(grid[0,0:])
ax2 = plt.subplot(grid[1,0:])
ax3 = plt.subplot(grid[2,0])
ax4 = plt.subplot(grid[2,1])
ax5 = plt.subplot(grid[3,0])
ax6 = plt.subplot(grid[3,1])


ax1.plot(np.arange(mpfc_act_np.shape[1]), mpfc_act_np[trial_pick][:,0], label='(A1,S1A1)')
ax1.plot(np.arange(mpfc_act_np.shape[1]), mpfc_act_np[trial_pick][:,1], label='(A1,S1A2)')
ax1.plot(np.arange(mpfc_act_np.shape[1]), mpfc_act_np[trial_pick][:,2], label='(A1,S2A1)')
ax1.plot(np.arange(mpfc_act_np.shape[1]), mpfc_act_np[trial_pick][:,3], label='(A1,S2A2)')
ax1.plot(np.arange(mpfc_act_np.shape[1]), mpfc_act_np[trial_pick][:,4], label='(A2,S1A1)')
ax1.plot(np.arange(mpfc_act_np.shape[1]), mpfc_act_np[trial_pick][:,5], label='(A2,S1A2)')
ax1.plot(np.arange(mpfc_act_np.shape[1]), mpfc_act_np[trial_pick][:,6], label='(A2,S2A1)')
ax1.plot(np.arange(mpfc_act_np.shape[1]), mpfc_act_np[trial_pick][:,7], label='(A2,S2A2)')

ax1.set_title('Activity of mPFC over time (trial {trial_pick})'.format(trial_pick=trial_pick))
ax1.set(xlabel='time (ms)', ylabel='firing rate')
ax1.legend(loc='upper right', title='Sequence')

vp_dir_trial_avg = np.empty((vp_act_np.shape[0], vp_act_np.shape[2]))
for idx, act in enumerate(vp_act_np):    
    vp_dir_trial_avg[idx] =  np.average(act, axis=0)

ax2.plot(np.arange(vp_dir_trial_avg.shape[0]), vp_dir_trial_avg[:,0], label='(A1,S1A1)')
ax2.plot(np.arange(vp_dir_trial_avg.shape[0]), vp_dir_trial_avg[:,1], label='(A1,S1A2)')
ax2.plot(np.arange(vp_dir_trial_avg.shape[0]), vp_dir_trial_avg[:,2], label='(A1,S2A1)')
ax2.plot(np.arange(vp_dir_trial_avg.shape[0]), vp_dir_trial_avg[:,3], label='(A1,S2A2)')
ax2.plot(np.arange(vp_dir_trial_avg.shape[0]), vp_dir_trial_avg[:,4], label='(A2,S1A1)')
ax2.plot(np.arange(vp_dir_trial_avg.shape[0]), vp_dir_trial_avg[:,5], label='(A2,S1A2)')
ax2.plot(np.arange(vp_dir_trial_avg.shape[0]), vp_dir_trial_avg[:,6], label='(A2,S2A1)')
ax2.plot(np.arange(vp_dir_trial_avg.shape[0]), vp_dir_trial_avg[:,7], label='(A2,S2A2)')

ax2.set_title('Mean activity of VP over trials')
ax2.set(xlabel='trial', ylabel='neuron activity')

# vp_ind_trial_avg = np.empty((vp_ind_act_np.shape[0], vp_ind_act_np.shape[2]))
# for idx, act in enumerate(vp_ind_act_np):    
#     vp_ind_trial_avg[idx] =  np.average(act, axis=0)
# ax3.plot(np.arange(vp_ind_trial_avg.shape[0]), vp_ind_trial_avg[:,0], label='(F1,B1)')
# ax3.plot(np.arange(vp_ind_trial_avg.shape[0]), vp_ind_trial_avg[:,1], label='(F1,B2)')
# ax3.plot(np.arange(vp_ind_trial_avg.shape[0]), vp_ind_trial_avg[:,2], label='(F2,S1)')
# ax3.plot(np.arange(vp_ind_trial_avg.shape[0]), vp_ind_trial_avg[:,3], label='(F2,S2)')
# ax3.plot(np.arange(vp_ind_trial_avg.shape[0]), vp_ind_trial_avg[:,4], label='(T1,B1)')
# ax3.plot(np.arange(vp_ind_trial_avg.shape[0]), vp_ind_trial_avg[:,5], label='(T1,B2)')
# ax3.plot(np.arange(vp_ind_trial_avg.shape[0]), vp_ind_trial_avg[:,6], label='(T2,S1)')
# ax3.plot(np.arange(vp_ind_trial_avg.shape[0]), vp_ind_trial_avg[:,7], label='(T2,S2)')

# ax3.set_title('Mean activity of VP ind over trials')
# ax3.set(xlabel='trial', ylabel='neuron activity')
# ax3.legend(loc='upper right', title='Sequence')

#FOR DAW TASK
# ax3.plot(np.arange(rewards_over_trial.shape[0]), rewards_over_trial[:,0], label='B1')
# ax3.plot(np.arange(rewards_over_trial.shape[0]), rewards_over_trial[:,1], label='B2')
# ax3.plot(np.arange(rewards_over_trial.shape[0]), rewards_over_trial[:,2], label='S1')
# ax3.plot(np.arange(rewards_over_trial.shape[0]), rewards_over_trial[:,3], label='S2')

# ax3.set_ylim([0.0, 1.0])
# ax3.set_title('Reward probability of second state option over time')
# ax3.set(xlabel='trial', ylabel='probability')
# ax3.legend(loc='upper right', title='Option')

print(rewards_over_trial)
ax3.plot(np.arange(rewards_over_trial.shape[0]), rewards_over_trial[:,0], label='S1A1', color='red')
ax3.legend(loc='upper right', title='Action')
ax3.set(ylabel='reward probability')
ax4.plot(np.arange(rewards_over_trial.shape[0]), rewards_over_trial[:,1], label='S1A2', color='green')
ax4.legend(loc='upper right', title='Action')
ax5.plot(np.arange(rewards_over_trial.shape[0]), rewards_over_trial[:,2], label='S2A1', color='blue')
ax5.set(xlabel='trial', ylabel='reward probability')
ax5.legend(loc='upper right', title='Action')
ax6.plot(np.arange(rewards_over_trial.shape[0]), rewards_over_trial[:,3], label='S2A2', color='orange')
ax6.legend(loc='upper right', title='Action')
ax6.set(xlabel='trial')

# ax3.set_ylim([0.0, 1.0])
# ax4.set_ylim([0.0, 1.0])
# ax5.set_ylim([0.0, 1.0])
# ax6.set_ylim([0.0, 1.0])
# ax3.set_title('Reward probability of second state option over time')
# ax3.set(xlabel='trial', ylabel='probability')
# ax3.legend(loc='upper right', title='Option')

plt.tight_layout()
plt.show()