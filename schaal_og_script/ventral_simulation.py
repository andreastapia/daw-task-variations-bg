from ANNarchy import *
import pylab as plt
import random
import sys
import scipy.spatial.distance
import pickle
import shutil
import os
import datetime

def saveAllActivations(foldername):    

    if not os.path.exists(foldername):
        os.makedirs(foldername)

##changes reward probability between trials
def change_reward_prob(current_value,mean,sd,lower_bound,upper_bound):
	shift=np.random.normal(mean,sd)
	new_value=current_value+shift

	#reflecting boundaries
	if new_value > upper_bound:
		new_value=2*upper_bound-new_value
	elif new_value < lower_bound:
		new_value=2*lower_bound-new_value

	return new_value



exec(open('schaal_og_script/ventral_model.py').read())

#enable learning for all projections
enable_learning()
random.seed()
num_trials = 300
eps=0.000001
stimuli_history=[]

#keys for the actions
stimulis=list(range(8))
stim_to_action={0:"(1,A)",1:"(1,B)",2:"(1,C)",3:"(1,D)",4:"(2,A)",5:"(2,B)",6:"(2,C)",7:"(2,D)"}

realizable_stimulus=[]
#stores info about the trials
common_transition=[]
rewarded=[]
rewards=[]
second_actions=[]

#every yellow underline variable come from the full_model.py file
Input_neurons.noise = 0.0 #0.001#0.001 

# set monitor variables for output of important activities, this records variables or populations
vp_activities=[]
monitor_vp=Monitor(VP_dir,'r')
monitor_vp.pause()
monitor_vp_ind=Monitor(VP_indir,'r')
monitor_vp_ind.pause()
vp_ind_activities=[]
objective_activities=[]
monitor_objectives=Monitor(Objectives,'r')
monitor_objectives.pause()

gpi_activities=[]
dPFC_activities=[]
gpe_activities=[]
strd1_activities=[]

monitor_objective_learning=Monitor(Objectives,"r")
monitor_objective_learning.pause()

# initialize some arrays for the results
objective_learning=[]
selected_objectives=[]
achieved_sequences=[]
reward_of_actions=[0.75, 0.25, 0.25, 0.25]
dopamine_activities=[]
achieved_desired_sequence=[]
acc_reward=[0]



# init the task transition probability 
transition_prob=0.7

for trial in range(num_trials):
    #reset everything at start of trial
    #ventral loop
    Input_neurons.baseline = 0.0
    Input_neurons.r = 0.0
    Objectives.r = 0.0
    Objectives.baseline = 0.0
    Thal_vent.r=0
    Thal_vent.baseline=0
    PPTN.baseline = 0

    simulate(2000)

    Thal_vent.baseline=1.2
    # reset finished

    monitor_objectives.resume()
    monitor_vp.resume()
    monitor_vp_ind.resume()
    simulate(200)   

    # show the stimuli in random order
    random.shuffle(stimulis)
    stimuli_history.append(np.array(stimulis))   

    for stim in stimulis:
        # show each for 300 ms
        Input_neurons[:,stim//4].baseline=1.0 
        Input_neurons[:,2+stim%4].baseline=1.0
        simulate(300)

        ## save activities on some trials
        #if trial in [0,10,20,40,60,80]:
        #    foldername=str(trial)+"data"+str(stim)
        #    saveAllActivations(foldername)         
        Input_neurons[:,stim//4].baseline=0    
        Input_neurons[:,2+stim%4].baseline=0
        simulate(10)



    # finished the sequence presentation
    Input_neurons.baseline=0 

    # short pause to check if WM retains sequence / has decided on one
    simulate(1000)

    # make the first decision according to Objective neurons
    selected_objective=random.choices(list(range(8)),weights=(Objectives.r+eps),k=1)[0]
    selected_objectives.append(selected_objective)

    #value [0,1] -> 1,2
    first_action = selected_objective//4
    #value [0,1,2,3] -> A,B,C,D
    second_action = selected_objective%4

    #change inputs to other panel, depending on the transition probability (see paper of Daw 2011)
    if first_action==0:
        if random.random()<transition_prob:
            second_panel=1
            common_transition.append(1)
        else:
            second_panel=2
            common_transition.append(0)

    if first_action==1:
        if random.random()<transition_prob:
            second_panel=2
            common_transition.append(1)
        else:
            second_panel=1
            common_transition.append(0)

    realizable = False
    if (second_panel==1 and (selected_objective in [0,1,4,5])) or (second_panel==2 and selected_objective in [2,3,6,7]):
        realizable_stimulus.append(True)
        realizable = True
    else:
        realizable_stimulus.append(False)

    #some time for second decision -> can be shorter if one wants that
    simulate(1000)

    #second decision
    if realizable == False:
        second_action = random.choices([0,1],k=1)[0]
        #second action is in the range [0,1,2,3] -> [A,B,C,D]
        if second_panel==2:
            second_action+=2

    second_actions.append(second_action)
    monitor_objectives.pause()
    monitor_vp.pause()
    monitor_vp_ind.pause()    

    objective_activities.append(monitor_objectives.get("r"))
    vp_activities.append(monitor_vp.get('r'))
    vp_ind_activities.append(monitor_vp_ind.get('r')) 

    achieved_sequence=4*first_action+second_action
    achieved_sequences.append(achieved_sequence)
    
    monitor_objective_learning.resume()    

    ## learning part
    # integration of the realizied objective into the cortex -> a bit weird as we we have to circumvent very high PFC activities (otherwise unlearning because of regularization with alpha)
    Objectives[achieved_sequence].baseline=2.0
    Input_neurons[:,first_action].baseline=0.5 
    Input_neurons[:,2+second_action].baseline=0.5

    # first short step with higer intergration
    simulate(100)

    # now reduce the feedback strenght
    Objectives[achieved_sequence].baseline=0.2
    StrThal_shell.learning=1

    #again, short time of cortical integration
    simulate(300)

    # task reward
    reward_random=random.random()
    acc_reward_new_entry=acc_reward[-1]

    rew = 0
    if reward_random < reward_of_actions[second_action]:
        PPTN.baseline=1.0
        acc_reward_new_entry+=1
        rewarded.append(1)
        rew = 1
    else:
        rewarded.append(0)
        acc_reward.append(acc_reward_new_entry)

    #allow weight changes

    ## if we want to delay learning in the ventral loop for the first 100 trials
    #if trial >= 100: 
    VTA.firing=1  

    #actual learning period
    simulate(100)

    PPTN.baseline=0.0
    VTA.firing=0
    StrThal_shell.learning=0    

    monitor_objective_learning.pause()

    objective_learning.append(monitor_objective_learning.get("r"))

    #save activities of important nuclei during certain trials 
    #if trial in [0,10,20,40,60,80]:
    #    foldername=str(trial)+"data8"
    #    saveAllActivations(foldername)

    dopamine_activities.append(VTA.r)
    achieved_desired_sequence.append(selected_objective==achieved_sequence)

    rewards.insert(trial, reward_of_actions.copy())

    print(
        "obj", selected_objective,
        "ach", achieved_sequence,
        "first",first_action, 
        "second",second_action,
        "realizable", realizable,
        "rewarded", rew,
        "trial", trial)



    

rewards=np.array(rewards)    
achieved_desired_sequence=np.array(achieved_desired_sequence)

now = datetime.datetime.now()

result_path = 'basal_ganglia/results/og_model_ventral/simulation_og_{date}/'.format(date=now.strftime("%Y%m%d%H%M%S"))
os.makedirs(result_path)
#np.save(result_path + "vp_activities.npy",vp_activities)
#np.save(result_path + "vp_ind_activities.npy",vp_ind_activities)
#np.save(result_path + "mpfc_activities.npy",objective_activities)
#np.save("stimuli_history.npy",stimuli_history)
#np.savetxt("results.txt",np.array(rewards))
#np.save("dopamine_activities.npy",dopamine_activities)

np.save(result_path + "rewards_over_time.npy", rewards)
np.save(result_path + "selected_objectives.npy",selected_objectives)
np.save(result_path + "achieved_sequences.npy",achieved_sequences)

acc_reward=np.array(acc_reward[1:])
np.save(result_path + "acc_reward.npy",acc_reward)

#np.save("gpi_activities.npy",gpi_activities)
#np.save("gpe_activities.npy",gpe_activities)
#np.save("dPFC_activities.npy",dPFC_activities)
#np.save("strd1_activities.npy",strd1_activities)
#np.save(result_path + "objective_learning.npy",objective_learning)
#np.save("gpi_learning.npy",gpi_learning)
#np.save("dpfc_learning.npy",dpfc_learning)

np.save(result_path + "common_transitions.npy",common_transition)
np.save(result_path + "rewarded.npy",rewarded)

#realizable_stimulus=np.array(realizable_stimulus)
#mask=[True if i else False for i in realizable_stimulus]

realizable_stimulus=np.array(realizable_stimulus)
np.save(result_path + "realizable_stimulus.npy",realizable_stimulus)





    

