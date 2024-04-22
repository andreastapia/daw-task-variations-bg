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

    # ventral
    np.savetxt(os.path.join(foldername,"hippo.txt"),Input_neurons.r)
    np.savetxt(os.path.join(foldername,"striatumdeins.txt"),StrD1_shell.r)
    np.savetxt(os.path.join(foldername,"striatumdzwei.txt"),StrD2_shell.r)
    np.savetxt(os.path.join(foldername,"STN.txt"),STN_shell.r)
    np.savetxt(os.path.join(foldername,"VPi.txt"),VP_dir.r)
    np.savetxt(os.path.join(foldername,"VPe.txt"),VP_indir.r)
    np.savetxt(os.path.join(foldername,"mPFC.txt"),Objectives.r)
    np.savetxt(os.path.join(foldername,"Thalamus.txt"),Thal_vent.r)
    np.savetxt(os.path.join(foldername,"VTA.txt"),VTA.r)
    np.savetxt(os.path.join(foldername,"RPE.txt"),RPE.r)
    np.savetxt(os.path.join(foldername,"PPTN.txt"),PPTN.r)
    np.savetxt(os.path.join(foldername,"CoreDeins.txt"),StrD1_core.r)
    np.savetxt(os.path.join(foldername,"SNr.txt"),SNr_core.r)
    np.savetxt(os.path.join(foldername,"feedback.txt"),StrThal_shell.r)
    np.savetxt(os.path.join(foldername,"RP.txt"),RPE.r)

    # dorsolateral
    np.savetxt(os.path.join(foldername,"VI.txt"),Visual_input.r)
    np.savetxt(os.path.join(foldername,"dPFC.txt"),dPFC.r)
    np.savetxt(os.path.join(foldername,"StrD1_caud.txt"),StrD1_caud.r)
    np.savetxt(os.path.join(foldername,"GPe.txt"),GPe.r)
    np.savetxt(os.path.join(foldername,"GPi.txt"),GPi.r)
    np.savetxt(os.path.join(foldername,"STN_caud.txt"),STN_caud.r)
    np.savetxt(os.path.join(foldername,"StrD2_caud.txt"),StrD2_caud.r)
    np.savetxt(os.path.join(foldername,"Thal_caud.txt"),Thal_caud.r)
    np.savetxt(os.path.join(foldername,"StrThal_caud.txt"),StrThal_caud.r)
    np.savetxt(os.path.join(foldername,"SNc_caud.txt"),SNc_caud.r)
    np.savetxt(os.path.join(foldername,"realized.txt"),Realized_sequence.r)    

    # dorsomedial
    np.savetxt(os.path.join(foldername,"StrD1_put.txt"),StrD1_put.r)
    np.savetxt(os.path.join(foldername,"GPi_put.txt"),GPi_put.r)
    np.savetxt(os.path.join(foldername,"Thal_put.txt"),Thal_put.r)
    np.savetxt(os.path.join(foldername,".txt"),Premotor.r)
    np.savetxt(os.path.join(foldername,"recency_bias_put.txt"),recency_bias_put.r)

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



exec(open('schaal_og_script/full_model.py').read())

#enable learning for all projections
enable_learning()
random.seed()
num_trials = 400
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

monitor_gpi=Monitor(GPi,"r")
monitor_gpi.pause()

monitor_gpe=Monitor(GPe,"r")
monitor_gpe.pause()

monitor_dPFC=Monitor(dPFC,"r")
monitor_dPFC.pause()

monitor_strd1=Monitor(StrD1_caud,"r")
monitor_strd1.pause()

monitor_objective_learning=Monitor(Objectives,"r")
monitor_objective_learning.pause()

monitor_gpi_learning=Monitor(GPi,"r")
monitor_gpi_learning.pause()

monitor_dpfc_learning=Monitor(dPFC,"r")
monitor_dpfc_learning.pause()


# initialize some arrays for the results
objective_learning=[]
gpi_learning=[]
dpfc_learning=[]
selected_objectives=[]
achieved_sequences=[]
reward_of_actions=[]
dopamine_activities=[]
achieved_desired_sequence=[]
acc_reward=[0]



# init the task transition probability 
transition_prob=0.7

# initialisation of reward distribtion
for i in range(4):
    reward_of_actions.append(np.random.uniform(0.25,0.75))

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

    #dorsomediamedial loop 
    Visual_input.baseline=0
    GPi.r=0
    GPe.r=0
    dPFC.r=0
    dPFC.baseline=0
    StrThal_caud.r=0
    Thal_caud.r=0
    StrD1_caud.r=0
    StrD2_caud.r=0
    Thal_caud.baseline=0.0
    Realized_sequence.baseline=0
    Realized_sequence.r=0
    StrThal_caud.learning=1.0
    simulate(2000)

    Thal_vent.baseline=1.15     
    Thal_caud.baseline=0.95
    Visual_input[:,0].baseline=1.5
    # reset finished

    monitor_objectives.resume()
    monitor_vp.resume()
    monitor_vp_ind.resume()
    monitor_gpi.resume()
    monitor_gpe.resume()
    monitor_dPFC.resume()
    monitor_strd1.resume()
    simulate(200)   

    # show the stimuli in random order
    random.shuffle(stimulis)
    stimuli_history.append(np.array(stimulis))   

    for stim in stimulis:
        # show each for 300 ms
        Input_neurons[:,stim//4].baseline=0.5 
        Input_neurons[:,2+stim%4].baseline=0.5
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
    simulate(200)

    # make the first decision according to Premotor neurons
    motor_objectives=Premotor.r
    first_action=random.choices([0,1],weights=motor_objectives+eps,k=1)[0]
    selected_objective=random.choices(list(range(8)),weights=(Objectives.r+eps),k=1)[0]
    selected_objectives.append(selected_objective)

    #change inputs to other panel, depending on the transition probability (see paper of Daw 2011)
    Visual_input.baseline=0.0

    if first_action==0:
        if random.random()<transition_prob:
            Visual_input[:,1].baseline=1.5
            second_panel=1
            common_transition.append(1)
        else:
            Visual_input[:,2].baseline=1.5
            second_panel=2
            common_transition.append(0)

    if first_action==1:
        if random.random()<transition_prob:
            Visual_input[:,2].baseline=1.5
            second_panel=2
            common_transition.append(1)
        else:
            Visual_input[:,1].baseline=1.5
            second_panel=1
            common_transition.append(0)

    if (second_panel==1 and (selected_objective in [0,1,4,5])) or (second_panel==2 and selected_objective in [2,3,6,7]):
        realizable_stimulus.append(True)
    else:
        realizable_stimulus.append(False)

    # decision of the Wm is reinforced, as no other sequences are shown by the hippocampus
    StrThal_caud.learning=1.0   

    #some time for second decision -> can be shorter if one wants that
    simulate(1000)

    #second decision
    motor_objectives=Premotor.r
    second_action=random.choices([0,1],weights=motor_objectives+eps,k=1)[0]

    if second_panel==2:
        second_action+=2

    second_actions.append(second_action)
    monitor_gpi.pause()
    monitor_gpe.pause()
    monitor_dPFC.pause()
    monitor_objectives.pause()
    monitor_strd1.pause()
    monitor_vp.pause()
    monitor_vp_ind.pause()    

    dPFC_activities.append(monitor_dPFC.get("r"))
    gpi_activities.append(monitor_gpi.get("r"))
    objective_activities.append(monitor_objectives.get("r"))
    gpe_activities.append(monitor_gpe.get("r"))
    strd1_activities.append(monitor_strd1.get("r"))
    vp_activities.append(monitor_vp.get('r'))
    vp_ind_activities.append(monitor_vp_ind.get('r')) 

    achieved_sequence=4*first_action+second_action
    achieved_sequences.append(achieved_sequence)
    
    monitor_objective_learning.resume()    
    monitor_gpi_learning.resume()
    monitor_dpfc_learning.resume()

    ## learning part
    # integration of the realizied objective into the cortex -> a bit weird as we we have to circumvent very high PFC activities (otherwise unlearning because of regularization with alpha)
    Realized_sequence[achieved_sequence].baseline=2.0
    Visual_input.baseline=0
    Input_neurons[:,first_action].baseline=0.5 
    Input_neurons[:,2+second_action].baseline=0.5
    dPFC[first_action].baseline=2.0
    dPFC[2+second_action].baseline=2.0

    # first short step with higer intergration
    simulate(100)

    # now reduce the feedback strenght
    dPFC[first_action].baseline=1.5
    dPFC[2+second_action].baseline=1.5
    StrThal_shell.learning=1
    Realized_sequence[achieved_sequence].baseline=0.2

    #again, short time of cortical integration
    simulate(300)

    # task reward
    reward_random=random.random()
    acc_reward_new_entry=acc_reward[-1]

    if reward_random < reward_of_actions[second_action]:
        PPTN.baseline=1.0
        acc_reward_new_entry+=1
        rewarded.append(1)
    else:
        rewarded.append(0)
        acc_reward.append(acc_reward_new_entry)

    #allow weight changes
    SNc_caud.firing=1

    ## if we want to delay learning in the ventral loop for the first 100 trials
    if trial >= 100: 
        VTA.firing=1  

    #actual learning period
    simulate(100)

    SNc_caud.firing=0
    PPTN.baseline=0.0
    VTA.firing=0
    StrThal_shell.learning=0    

    monitor_objective_learning.pause()
    monitor_gpi_learning.pause()
    monitor_dpfc_learning.pause()    

    objective_learning.append(monitor_objective_learning.get("r"))
    gpi_learning.append(monitor_gpi_learning.get("r"))
    dpfc_learning.append(monitor_dpfc_learning.get("r"))

    

    #save activities of important nuclei during certain trials 
    #if trial in [0,10,20,40,60,80]:
    #    foldername=str(trial)+"data8"
    #    saveAllActivations(foldername)

    dopamine_activities.append(VTA.r)
    achieved_desired_sequence.append(selected_objective==achieved_sequence)

    print(selected_objective)
    rewards.append(selected_objective==0)

    # change the rewards
    for i in range(4):
        reward_of_actions[i]=change_reward_prob(reward_of_actions[i],0,0.025,0.25,0.75)

    

    

rewards=np.array(rewards)    
achieved_desired_sequence=np.array(achieved_desired_sequence)

now = datetime.datetime.now()

result_path = 'basal_ganglia/results/og_model/simulation_og_{date}/'.format(date=now.strftime("%Y%m%d%H%M%S"))
os.makedirs(result_path)
#np.save("vp_activities.npy",vp_activities)
#np.save("vp_ind_activities.npy",vp_ind_activities)
#np.save("objective_activities.npy",objective_activities)
#np.save("stimuli_history.npy",stimuli_history)
#np.savetxt("results.txt",np.array(rewards))
#np.save("dopamine_activities.npy",dopamine_activities)


np.save(result_path + "selected_objectives.npy",selected_objectives)
np.save(result_path + "achieved_sequences.npy",achieved_sequences)

acc_reward=np.array(acc_reward[1:])
np.save(result_path + "acc_reward.npy",acc_reward)

#np.save("gpi_activities.npy",gpi_activities)
#np.save("gpe_activities.npy",gpe_activities)
#np.save("dPFC_activities.npy",dPFC_activities)
#np.save("strd1_activities.npy",strd1_activities)
#np.save("objective_learning.npy",objective_learning)
#np.save("gpi_learning.npy",gpi_learning)
#np.save("dpfc_learning.npy",dpfc_learning)

np.save(result_path + "common_transitions.npy",common_transition)
np.save(result_path + "rewarded.npy",rewarded)

#realizable_stimulus=np.array(realizable_stimulus)
#mask=[True if i else False for i in realizable_stimulus]

realizable_stimulus=np.array(realizable_stimulus)
np.save(result_path + "realizable_stimulus.npy",realizable_stimulus)





    

