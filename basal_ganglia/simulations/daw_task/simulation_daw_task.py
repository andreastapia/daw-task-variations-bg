from ANNarchy import Monitor, enable_learning, simulate
from models.daw_task.projections import *
from utils.utils import *
import numpy as np
import pylab as plt
import random
import os
import datetime


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

def diffuse_prob(prob):
    """Diffuses a probability between 0.25 and 0.75"""
    prob += random.gauss(0, 0.025)
    if prob < 0.25:
        prob = 0.5 - prob
    elif prob > 0.75:
        prob = 1.5 - prob
    assert prob >= 0.25 and prob <= 0.75
    return prob

def daw_task():
    now = datetime.datetime.now()
    result_path = 'basal_ganglia/results/daw_task_bigger_pretrain/simulation_{date}/'.format(date=now.strftime("%Y%m%d%H%M%S"))
    os.makedirs(result_path)

    #enable learning for all projections
    enable_learning()
    random.seed()
    num_trials = 450
    eps=0.000001


    stimulis=list(range(8))
    stim_to_action={
        0:"(1,A)",
        1:"(1,B)",
        2:"(1,C)",
        3:"(1,D)",
        4:"(2,A)",
        5:"(2,B)",
        6:"(2,C)",
        7:"(2,D)"}

    stimuli_history=[]
    realizable_stimulus=[]
    #stores info about the trials
    common_transition=[]
    rewarded=[]
    rewards=[]
    second_actions=[]

    # set monitor variables for output of important activities, this records variables or populations
    monitor_dorso_thal=Monitor(DorsomedialBG_Thal,'r')
    monitor_dorso_thal.pause()
    monitor_gpi=Monitor(GPi,'r')
    monitor_gpi.pause()
    monitor_dpfc=Monitor(dPFC,'r')
    monitor_dpfc.pause()
    monitor_strd1=Monitor(DorsomedialBG_StrD1,'r')
    monitor_strd1.pause()
    monitor_strd1_learning=Monitor(DorsomedialBG_StrD1,'r')
    monitor_strd1_learning.pause()
    monitor_feedback=Monitor(DorsomedialBG_cortical_feedback,'r')
    monitor_feedback.pause()
    monitor_gpi_learning=Monitor(GPi,'r')
    monitor_gpi_learning.pause()
    monitor_dpfc_learning=Monitor(dPFC,'r')
    monitor_dpfc_learning.pause()


    monitor_vp=Monitor(VentralBG_VP_dir,'r')
    monitor_vp.pause()
    monitor_vp_ind=Monitor(VentralBG_VP_indir,'r')
    monitor_vp_ind.pause()
    monitor_mpfc=Monitor(mPFC,'r')
    monitor_mpfc.pause()
    monitor_Hippo=Monitor(Hippocampus,'r')
    monitor_Hippo.pause()
    monitor_naccd1=Monitor(NAccD1_shell,'r')
    monitor_naccd1.pause()
    monitor_naccd2=Monitor(NAccD2_shell,'r')
    monitor_naccd2.pause()
    monitor_thal=Monitor(VentralBG_Thal,'r')
    monitor_thal.pause()
    monitor_objective_learning=Monitor(mPFC,"r")
    monitor_objective_learning.pause()

    gpi_activities=[]
    dPFC_activities=[]
    dorso_thal_activities=[]
    strd1_activities=[]
    gpi_learning=[]
    dpfc_learning=[]
    strd1_learning=[]
    feedback_activities=[]

    vp_activities=[]
    vp_ind_activities=[]
    mpfc_activities=[]
    hippo_activities=[]
    thal_activities=[]
    objective_learning=[]
    naccd1_activities=[]
    naccd2_activities=[]

    # initialize some arrays for the results
    selected_objectives=[]
    achieved_sequences=[]
    reward_of_actions=[]
    dopamine_activities_vta=[]
    dopamine_activities_snc=[]
    achieved_desired_sequence=[]
    acc_reward=[0]

    Hippocampus.noise = 0.0 
    selected_obj_counter = 0
    transition_prob=0.7

    # initialisation of reward distribtion
    for i in range(4):
        reward_of_actions.append(np.random.uniform(0.25,0.75))

    for trial in range(num_trials):
        #reset everything at start of trial
        #ventral loop
        Hippocampus.baseline = 0.0
        Hippocampus.r = 0.0
        mPFC.r = 0.0
        mPFC.baseline = 0.0
        VentralBG_Thal.r=0
        VentralBG_Thal.baseline=0
        VentralBG_PPTN.baseline = 0
        VentralBG_Cortical_feedback.r = 0

        #dorsomediamedial loop 
        Visual_input.baseline=0
        GPi.r=0
        GPe.r=0
        dPFC.r=0
        dPFC.baseline=0
        DorsomedialBG_cortical_feedback.r=0
        DorsomedialBG_Thal.r=0
        DorsomedialBG_StrD1.r=0
        DorsomedialBG_StrD2.r=0
        DorsomedialBG_Thal.baseline=0.0
        DorsomedialBG_PPTN.baseline=0
        DorsomedialBG_PPTN.r=0
        DorsomedialBG_cortical_feedback.learning=1.0
        #intertrial
        simulate(2000)

        # reset finished
        monitor_dorso_thal.resume()
        monitor_gpi.resume()
        monitor_dpfc.resume()
        monitor_strd1.resume()
        monitor_feedback.resume()
        monitor_mpfc.resume()
        monitor_vp.resume()
        monitor_vp_ind.resume()
        monitor_thal.resume()
        monitor_Hippo.resume()
        monitor_naccd1.resume()
        monitor_naccd2.resume() 

        VentralBG_Thal.baseline=1.15
        DorsomedialBG_Thal.baseline=0.95
        Visual_input[:,0].baseline=1.5

        simulate(200)   

        # show the stimuli in random order
        random.shuffle(stimulis)
        stimuli_history.append(np.array(stimulis))   

        # show each stimuli for 300 ms
        # stimuli is represented by an integer eg. (3,4,7,0,1,5,2,6)
        # check with sim_to_action
        for stim in stimulis:
            Hippocampus[:,stim//4].baseline=0.5 
            Hippocampus[:,2+stim%4].baseline=0.5
            #simulate for the action, eg. first iteration [:,0] , [:,5] -> represents action (1,D)
            #all ather baselines are set to zero
            simulate(300)
            ## save activities on some trials
            #if trial in [0,10,20,40,60,80]:
            #    foldername=str(trial)+"data"+str(stim)
            #    saveAllActivations(foldername)         
            Hippocampus[:,stim//4].baseline=0    
            Hippocampus[:,2+stim%4].baseline=0
            simulate(10)

        # finished the sequence presentation
        Hippocampus.baseline=0 

        monitor_Hippo.pause()
        monitor_naccd1.pause()
        monitor_naccd2.pause() 

        # short pause to check if WM retains sequence / has decided on one
        simulate(200)

        #objective action depending on mPFC, these represents same transition as sim_to_action
        selected_objective=random.choices(list(range(8)),weights=(mPFC.r+eps),k=1)[0]
        selected_objectives.append(selected_objective)

        # make the first decision according to Premotor neurons
        motor_objectives=Premotor.r
        #first action selected: left or right -> (0,1)
        first_action=random.choices([0,1],weights=motor_objectives+eps,k=1)[0]

        #change inputs to other panel, depending on the transition probability (see paper of Daw 2011)
        Visual_input.baseline=0.0

        #action 1
        if first_action==0:
            #common transition
            if random.random()<transition_prob:
                Visual_input[:,1].baseline=1.5
                second_panel=1
                common_transition.append(1)
            #rare transition
            else:
                Visual_input[:,2].baseline=1.5
                second_panel=2
                common_transition.append(0)
        #action 2
        if first_action==1:
            #common transition
            if random.random()<transition_prob:
                Visual_input[:,2].baseline=1.5
                second_panel=2
                common_transition.append(1)
            #rare transition
            else:
                Visual_input[:,1].baseline=1.5
                second_panel=1
                common_transition.append(0)

        # second_panel = 1 -> (A,B)
        # second_panel = 2 -> (C,D)
        #{0:"(1,A)",1:"(1,B)",2:"(1,C)",3:"(1,D)",4:"(2,A)",5:"(2,B)",6:"(2,C)",7:"(2,D)"}
        #check if stimulis proposed by the mPFC is doable
        #not doable when the objective isn't in the second panel
        #if second panel is 2 (C,D) sequences 0,1,4,5 are not possible
        realizable = False
        if (second_panel==1 and (selected_objective in [0,1,4,5])) or (second_panel==2 and selected_objective in [2,3,6,7]):
            realizable_stimulus.append(True)
            realizable = True
        else:
            realizable_stimulus.append(False)

        DorsomedialBG_cortical_feedback.learning=1.0  

        #some time for second decision -> can be shorter if one wants that
        simulate(1000)

        #you can save this as a separate file
        #second decision, choose between options from the second panel (left or right, same as before)
        motor_objectives=Premotor.r
        second_action=random.choices([0,1],weights=motor_objectives+eps,k=1)[0]

        #second action is in the range [0,1,2,3] -> [A,B,C,D]
        if second_panel==2:
            second_action+=2

        second_actions.append(second_action)
        #calculate the sequence selected so it's in the range [0,7]
        #eg. if sequence is 5 -> (2,B) this means that first and second action are 1 and 1 respectively
        #that gives a total of 4*1*+1 = 5, you can calculate the others to check
        achieved_sequence=4*first_action+second_action
        achieved_sequences.append(achieved_sequence)
        
        monitor_objective_learning.resume()
        monitor_gpi_learning.resume()
        monitor_dpfc_learning.resume()
        monitor_strd1_learning.resume()
        ## LEARNING PART
        # integration of the realizied objective into the cortex -> a bit weird as we we have to circumvent very high PFC activities (otherwise unlearning because of regularization with alpha)
        #update columns for the selected sequence

        Visual_input.baseline=0
        DorsomedialBG_PPTN[achieved_sequence].baseline=2.0
        Hippocampus[:,first_action].baseline=0.5 
        Hippocampus[:,2+second_action].baseline=0.5
        dPFC[first_action].baseline=2.0
        dPFC[2+second_action].baseline=2.0

        # first short step with higer intergration
        simulate(100)

        # now reduce the feedback strenght
        dPFC[first_action].baseline=1.5
        dPFC[2+second_action].baseline=1.5 
        VentralBG_Cortical_feedback.learning=1
        DorsomedialBG_PPTN[achieved_sequence].baseline=0.2

        #again, short time of cortical integration
        simulate(300)

        # task reward
        reward_random=random.random()
        #get last item form array
        #accumulated reward over iterations
        acc_reward_new_entry=acc_reward[-1]
        #this array is initialized before, check if the second action has a reward
        #rewarded
        rew = 0
        if reward_random < reward_of_actions[second_action]:
            VentralBG_PPTN.baseline=1.0
            acc_reward_new_entry+=1
            rewarded.append(1)
            rew = 1
        #non-rewarded
        else:
            rewarded.append(0)
            acc_reward.append(acc_reward_new_entry)

        DorsomedialBG_SNc.firing=1

        ## if we want to delay learning in the ventral loop for the first 100 trials
        if trial >= 150: 
            VTA.firing=1  

        #actual learning period, outcome
        simulate(100)

        DorsomedialBG_SNc.firing=0
        VentralBG_PPTN.baseline=0.0
        VTA.firing=0
        VentralBG_Cortical_feedback.learning=0    

        monitor_vp.pause()
        monitor_vp_ind.pause()    
        monitor_dorso_thal.pause()
        monitor_objective_learning.pause()
        monitor_gpi_learning.pause()
        monitor_dpfc_learning.pause()
        monitor_strd1_learning.pause()
        monitor_mpfc.pause()
        monitor_thal.pause()
        monitor_gpi.pause()
        monitor_dpfc.pause()
        monitor_strd1.pause()
        monitor_feedback.pause()

        thal_activities.append(monitor_thal.get("r"))
        vp_activities.append(monitor_vp.get('r'))
        vp_ind_activities.append(monitor_vp_ind.get('r')) 
        naccd1_activities.append(monitor_naccd1.get('r')) 
        naccd2_activities.append(monitor_naccd2.get('r')) 
        hippo_activities.append(monitor_Hippo.get("r"))
        dopamine_activities_vta.append(VTA.r)

        dPFC_activities.append(monitor_dpfc.get("r"))
        gpi_activities.append(monitor_gpi.get("r"))
        dorso_thal_activities.append(monitor_dorso_thal.get("r"))
        mpfc_activities.append(monitor_mpfc.get("r"))
        strd1_activities.append(monitor_strd1.get("r"))
        gpi_learning.append(monitor_gpi_learning.get("r"))
        dpfc_learning.append(monitor_dpfc_learning.get("r"))
        objective_learning.append(monitor_objective_learning.get("r"))
        strd1_learning.append(monitor_strd1_learning.get("r"))
        dopamine_activities_snc.append(DorsomedialBG_SNc.r)
        feedback_activities.append(monitor_feedback.get("r"))
        
        #this stores if the sequence proposed by the mPFC is actually selected by the other loops
        #this is not sored, you could
        achieved_desired_sequence.append(selected_objective==achieved_sequence)

        rewards.insert(trial, reward_of_actions.copy())

        if selected_objective==achieved_sequence:
            selected_obj_counter += 1
            
        print(
        "obj", selected_objective,
        "ach", achieved_sequence,
        "first",first_action, 
        "second",second_action,
        "rewarded", rew,
        "trial", trial)

        #change the rewards
        for i in range(4):
            reward_of_actions[i]=diffuse_prob(reward_of_actions[i])


    print(
    "total objective reached", selected_obj_counter,
        "out of", sum(realizable_stimulus),
        "meaning", (selected_obj_counter/sum(realizable_stimulus))*100,"%")
    rewards=np.array(rewards)    
    achieved_desired_sequence=np.array(achieved_desired_sequence)
    realizable_stimulus=np.array(realizable_stimulus)

    # np.save(result_path + "mpfc_activities.npy",mpfc_activities)
    # np.save(result_path + "objective_learning.npy",objective_learning)
    # np.save(result_path + "dopamine_activities_vta.npy",dopamine_activities_vta)
    # np.save(result_path + "thal_activities.npy",thal_activities)    
    # np.save(result_path + "vp_ind_activities.npy",vp_ind_activities)
    # np.save(result_path + "vp_activities.npy",vp_activities)
    # np.save(result_path + "naccd1_activities.npy",naccd1_activities)
    # np.save(result_path + "naccd2_activities.npy",naccd2_activities)
    # np.save(result_path + "hippo_activities.npy",hippo_activities)

    # np.save(result_path + "gpi_activities.npy",gpi_activities)
    # np.save(result_path + "dPFC_activities.npy",dPFC_activities)
    # np.save(result_path + "strd1_activities.npy",strd1_activities)
    # np.save(result_path + "strd1_learning.npy",strd1_learning)
    # np.save(result_path + "gpi_learning.npy",gpi_learning)
    # np.save(result_path + "dpfc_learning.npy",dpfc_learning)
    # np.save(result_path + "dopamine_activities_snc.npy",dopamine_activities_snc)
    # np.save(result_path + "feedback_activities.npy",feedback_activities)
    # np.save(result_path + "dorso_thal_activities.npy",dorso_thal_activities)

    np.save(result_path + "achieved_desired_sequence.npy",achieved_desired_sequence)
    np.save(result_path + "rewards_over_time.npy", rewards)
    np.save(result_path + "selected_objectives.npy",selected_objectives)
    np.save(result_path + "achieved_sequences.npy",achieved_sequences)
    acc_reward=np.array(acc_reward[1:])
    np.save(result_path + "acc_reward.npy",acc_reward)
    np.save(result_path + "common_transitions.npy",common_transition)
    np.save(result_path + "rewarded.npy",rewarded)
    np.save(result_path + "realizable_stimulus.npy",realizable_stimulus)




    

