from ANNarchy import Monitor, enable_learning, simulate
from models.doll_task.projections import *
# TODO:  try to do something with this import: from utils.utils import *
import numpy as np
import random
import os
import datetime

#sequences (objectives):
#F: face, B: body, T:tool, S:scene
#F1 -> B1, F1 -> B2
#F2 -> S1, F2 -> S2
#T1 -> B1, T1 -> B2
#T2 -> S1, T2 -> S2
def get_sequence(x, y):
    #print("first", x, "second", y)
    if x == 0 or x == 1:
        return y
    if x == 2:
        return 2*x+y
    if x == 3:
        return x+y+1

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


def doll_task():
    now = datetime.datetime.now()
    result_path = 'basal_ganglia/results/doll_task/simulation_{date}/'.format(date=now.strftime("%Y%m%d%H%M%S"))
    os.makedirs(result_path)

    #enable learning for all projections
    enable_learning()
    random.seed()
    num_trials = 400
    eps=0.000001

    #keys for the actions
    stimulis=list(range(8))
    #sequences (objectives):
    #F: face, B: body, T:tool, S:scene
    #F1 -> B1, F1 -> B2
    #F2 -> S1, F2 -> S2
    #T1 -> B1, T1 -> B2
    #T2 -> S1, T2 -> S2
    stim_to_action={
        0:"(F1,B1)",1:"(F1,B2)",2:"(F2,S1)",
        3:"(F2,S2)",4:"(T1,B1)",5:"(T1,B2)",
        6:"(T2,S1)",7:"(T2,S2)"}

    possible_first_actions = [0,1]
    stimuli_history=[]
    realizable_stimulus=[]
    #stores info about the trials
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
    # initialisation of reward distribtion
    for i in range(2):
        reward_of_actions.append(np.random.uniform(0.25,0.75))

    reward_of_actions.append(reward_of_actions[1])
    reward_of_actions.append(reward_of_actions[0])

    #iterations done by one person
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
        #determines if panel choices are swapped
        first_swapped = random.choices([0,1], k=1)[0]
        #chose random first panel and activate visual input accordingly
        #first choices are written like that for visual only, choices depends on premotor at the end
        first_panel = random.choices([0,1], k=1)[0]
        if first_panel == 0 and first_swapped == 0:
            Visual_input[:,0].baseline=1.5
        elif first_panel == 0 and first_swapped == 1:
            Visual_input[:,1].baseline=1.5
        elif first_panel == 1 and first_swapped == 0:
            Visual_input[:,2].baseline=1.5
        elif first_panel == 1 and first_swapped == 1:
            Visual_input[:,3].baseline=1.5
        # reset finished

        simulate(200)   

        #show inly possible sequences
        stimulis=list(range(8))
        first_panel = random.choices(possible_first_actions, k=1)[0]
        if first_panel == 0:
            stimulis = stimulis[:4]
        elif first_panel == 1:
            stimulis = stimulis[4:]

        # show the stimuli in random order
        random.shuffle(stimulis)
        stimuli_history.append(np.array(stimulis))   

        # show each stimuli for 300 ms
        # stimuli is represented by an integer eg. (3,4,7,0,1,5,2,6)
        # check with sim_to_action
        for stim in stimulis:   
            #updated division for doll task, logic is the same as daw task
            Hippocampus[:,stim//2].baseline=0.5 
            Hippocampus[:,4+stim%4].baseline=0.5
            # all ather baselines are set to zero
            simulate(300)    
            Hippocampus[:,stim//2].baseline=0    
            Hippocampus[:,4+stim%4].baseline=0
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
        #premotor gives you left or right choice, decode first action after
        pre_first_action=random.choices([0,1],weights=motor_objectives+eps,k=1)[0]
        first_action = -1
        # print(first_swapped, pre_first_action, first_panel)
        if first_swapped == 0 and pre_first_action == 0 and first_panel == 0:
            first_action = 0

        if first_swapped == 0 and pre_first_action == 1 and first_panel == 0:
            first_action = 1

        if first_swapped == 1 and pre_first_action == 0 and first_panel == 0:
            first_action = 1

        if first_swapped == 1 and pre_first_action == 1 and first_panel == 0:
            first_action = 0

        if first_swapped == 0 and pre_first_action == 0 and first_panel == 1:
            first_action = 2

        if first_swapped == 0 and pre_first_action == 1 and first_panel == 1:
            first_action = 3

        if first_swapped == 1 and pre_first_action == 0 and first_panel == 1:
            first_action = 3

        if first_swapped == 1 and pre_first_action == 1 and first_panel == 1:
            first_action = 2

        #if the selected panel is different from the objective 
        #it is not possible to do it
        realizable = False
        if (
            (first_action == 0 and selected_objective in [4,5,6,7]) or
            (first_action == 1 and selected_objective in [4,5,6,7]) or
            (first_action == 2 and selected_objective in [0,1,2,3]) or
            (first_action == 3 and selected_objective in [0,1,2,3])):
            realizable_stimulus.append(False)
        else:
            realizable_stimulus.append(True)
            realizable = True

        #change inputs to other panel, depending on the transition probability (see paper of Daw 2011)
        Visual_input.baseline=0.0

        # first action in range [0,1,2,3]
        # 0: face1, 1: face2, 2: tool1, 3:tool2
        #action F1
        #[0,2] -> second panel 1
        #[1,3] -> second panel 2
        #present it swapped if necessary
        second_swapped = random.choices([0,1], k=1)[0]
        #action F1
        second_panel = 0
        if first_action==0:
            if second_swapped == 0:
                Visual_input[:,4].baseline=1.5
            elif second_swapped == 1:
                Visual_input[:,5].baseline=1.5
            second_panel=1

        #action F2
        if first_action==1:
            if second_swapped == 0:
                Visual_input[:,6].baseline=1.5
            elif second_swapped == 1:
                Visual_input[:,7].baseline=1.5
            second_panel=2

        #action T1
        if first_action==2:
            if second_swapped == 0:
                Visual_input[:,4].baseline=1.5
            elif second_swapped == 1:
                Visual_input[:,5].baseline=1.5
            second_panel=1

        #action T2
        if first_action==3:
            if second_swapped == 0:
                Visual_input[:,6].baseline=1.5
            elif second_swapped == 1:
                Visual_input[:,7].baseline=1.5
            second_panel=2


        # decision of the Wm is reinforced, as no other sequences are shown by the hippocampus
        DorsomedialBG_cortical_feedback.learning=1.0   

        #some time for second decision -> can be shorter if one wants that
        simulate(1000)

        #second decision, choose between options from the second panel (left or right, same as before)
        motor_objectives=Premotor.r
        pre_second_action=random.choices([0,1],weights=motor_objectives+eps,k=1)[0]

        #second action is in the range [0,1,2,3] -> [B1,B2,S1,S2]
        #decode second action
        second_action = -1
        if second_swapped == 0 and pre_second_action == 0 and second_panel == 1:
            second_action = 0

        if second_swapped == 0 and pre_second_action == 1 and second_panel == 1:
            second_action = 1

        if second_swapped == 1 and pre_second_action == 0 and second_panel == 1:
            second_action = 1

        if second_swapped == 1 and pre_second_action == 1 and second_panel == 1:
            second_action = 0

        if second_swapped == 0 and pre_second_action == 0 and second_panel == 2:
            second_action = 2

        if second_swapped == 0 and pre_second_action == 1 and second_panel == 2:
            second_action = 3

        if second_swapped == 1 and pre_second_action == 0 and second_panel == 2:
            second_action = 3

        if second_swapped == 1 and pre_second_action == 1 and second_panel == 2:
            second_action = 2

        #you can save this as a separate file
        second_actions.append(second_action)

        achieved_sequence=get_sequence(first_action, second_action)
        achieved_sequences.append(achieved_sequence)

        monitor_objective_learning.resume()
        monitor_gpi_learning.resume()
        monitor_dpfc_learning.resume()
        monitor_strd1_learning.resume()

        ## LEARNING PART
        # integration of the realizied objective into the cortex -> a bit weird as we we have to circumvent very high PFC activities (otherwise unlearning because of regularization with alpha)
        Visual_input.baseline=0
        DorsomedialBG_PPTN[achieved_sequence].baseline=2.0
        Hippocampus[:,first_action].baseline=0.5 
        Hippocampus[:,4+second_action].baseline=0.5
        dPFC[first_action].baseline=2.0
        dPFC[4+second_action].baseline=2.0

        # first short step with higer intergration
        simulate(100)

        # now reduce the feedback strenght
        dPFC[first_action].baseline=1.5
        dPFC[4+second_action].baseline=1.5
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

        #allow weight changes
        DorsomedialBG_SNc.firing=1

        ## if we want to delay learning in the ventral loop for the first 100 trials
        if trial >= 100: 
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
            "realizable", realizable,
            "stimulis", stimulis,
            "trial", trial)

        # change the rewards
        # change the rewards
        for i in range(2):
            reward_of_actions[i]=change_reward_prob(reward_of_actions[i],0,0.025,0.25,0.75)

        reward_of_actions[2] = reward_of_actions[1]
        reward_of_actions[3] = reward_of_actions[0]

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
    np.save(result_path + "rewarded.npy",rewarded)
    np.save(result_path + "realizable_stimulus.npy",realizable_stimulus)



    

