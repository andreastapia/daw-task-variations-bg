from ANNarchy import (
    Population,
    Neuron,
    Projection,
    Synapse,
    Monitor,
    Normal,
    Uniform,
    enable_learning,
    simulate,
    compile
)
import random
import optuna
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error

#General networks parameters
#Populations setups
num_objectives=8
baseline_dopa_vent = 0.15
gen_tau_trace=100

#shared between populations and projections
baseline_dopa_caud=0.1
num_stim = 6

#Projections exclusive
# caudate
weight_local_inh = 0.8 *0.2 * 0.166 #0.02656 #old model (Scholl) with 6 distinct population -> 0.8*0.2, here only one -> scale it by 1/6
weight_stn_inh = 0.3 
weight_inh_sd2 = 0.5 *0.2
weight_inh_thal = 0.5 *0.2
modul = 0.01


#NEURONS
#Neuron models
#non-dopaminergic neurons
LinearNeuron = Neuron(
    parameters= """
        tau = 10.0
        baseline = 0.0
        noise = 0.0
        lesion = 1.0
    """,
    equations="""
        tau*dmp/dt + mp = sum(exc) - sum(inh) + baseline + noise*Uniform(-1.0,1.0)
        r = lesion*pos(mp)
    """
)

#non-dopaminergic with trace
LinearNeuron_trace = Neuron(
    parameters= """
        tau = 10.0
        baseline = 0.0
        noise = 0.0
        tau_trace = 120.0
        lesion = 1.0
    """,
    equations="""
        tau*dmp/dt + mp = sum(exc) - sum(inh) + baseline + noise*Uniform(-1.0,1.0)
        r = lesion*pos(mp)
        tau_trace*dtrace/dt + trace = r
    """
)

InputNeuron = Neuron(
    parameters="""
        tau = 1.5
        baseline = 0.0
        noise = 0.0
    """,
    equations="""
        tau*dmp/dt + mp = baseline + noise*Uniform(-1.0,1.0)
	    r = if (mp>0.0): mp else: 0.0
    """
)

LinearNeuron_learning = Neuron(
    parameters= """
        tau = 10.0
        baseline = 0.0
        noise = 0.0
        lesion = 1.0
        learning=0.0
    """,
    equations="""
        cortex_input= if learning==1.0: sum(exc) else: 0 
        tau*dmp/dt + mp = cortex_input - sum(inh) + baseline + noise*Uniform(-1.0,1.0)
        r = lesion*pos(mp)
    """
)

DopamineNeuron = Neuron(

    parameters="""
        tau = 10.0
        firing = 0
        inhibition = 0.0
        baseline = 0.0
        exc_threshold = 0.0
        factor_inh = 10.0
    """,
    equations="""
        ex_in = if (sum(exc)>exc_threshold): 1 else: 0
        s_inh = sum(inh)
        aux = if (firing>0): (ex_in)*(pos(1.0-baseline-s_inh) + baseline) + (1-ex_in)*(-factor_inh*sum(inh)+baseline)  else: baseline
        tau*dmp/dt + mp =  aux
        r = if (mp>0.0): mp else: 0.0
    """
)

# Neurons for the ventral loop, previously called "Objectives"
#mPFC, is shared with working memory and ventral loop
mPFC = Population(name="mPFC_Objectives", geometry=num_objectives, neuron=LinearNeuron)
mPFC.tau = 40.0 # 40
mPFC.noise = 0.01

# Hippocampus is implemented as Input neurons cycling through the different sequences 
# Used on the ventral loop
# previously called Input_Neurons
Hippocampus = Population(name='Hipp_Input_Neurons',geometry=(4,num_stim),neuron=InputNeuron)
Hippocampus.tau = 10.0
Hippocampus.baseline = 0.0
Hippocampus.noise = 0.0

# Nacc (Nucleus Accumbens) Shell, D1-type striatum neurons
# previously called StrD1_shell
NAccD1_shell = Population(name="NAcc_StrD1_shell", geometry=(4, num_stim),neuron = LinearNeuron_trace)
NAccD1_shell.tau = 10.0
NAccD1_shell.noise = 0.3
NAccD1_shell.baseline = 0.0
NAccD1_shell.lesion = 1.0
NAccD1_shell.tau_trace = gen_tau_trace

# Nacc (Nucleus Accumbens) Shell, D2-type striatum neurons, previously called StrD2_shell
# shell indirect pathway 
# similar to the direct
NAccD2_shell = Population(name="NAcc_StrD2_shell", geometry = (4, num_stim), neuron=LinearNeuron_trace)
NAccD2_shell.tau = 10.0
NAccD2_shell.noise = 0.01
NAccD2_shell.baseline = 0.0
NAccD2_shell.lesion = 1.0
NAccD2_shell.tau_trace = gen_tau_trace #this line had a duplicated trace for NAccD1_Shell


# shell VP (direct)
VentralBG_VP_dir = Population(name="VP_dir", geometry = num_objectives, neuron=LinearNeuron_trace)
VentralBG_VP_dir.tau = 10.0
VentralBG_VP_dir.noise = 0.3 #0.3
VentralBG_VP_dir.baseline = 1.5 #0.75 #1.5
VentralBG_VP_dir.tau_trace = gen_tau_trace

# shell VP (indirect)
VentralBG_VP_indir = Population(name="VP_indir", geometry = num_objectives, neuron=LinearNeuron_trace)
VentralBG_VP_indir.tau = 10.0
VentralBG_VP_indir.noise = 0.05
VentralBG_VP_indir.baseline = 1.0
VentralBG_VP_indir.tau_trace = gen_tau_trace #add trace for VP_dir?

# STN
VentralBG_STN = Population(name="VentralBG_STN_shell", geometry = (4, num_stim), neuron=LinearNeuron_trace)
VentralBG_STN.tau = 10.0
VentralBG_STN.noise = 0.01
VentralBG_STN.baseline = 0.0
VentralBG_STN.tau_trace = gen_tau_trace #add trace for STN Shell?

# Feedback from Cortex/Thal to GPi
# enables hebbian learning rule by excitation of the VP neurons
# previously called StrThal_shell
VentralBG_Cortical_feedback = Population(name="VentralBG_Cortical_feedback", geometry = num_objectives, neuron=LinearNeuron_learning)
VentralBG_Cortical_feedback.tau = 10.0
VentralBG_Cortical_feedback.noise = 0.01
VentralBG_Cortical_feedback.baseline = 0.4
VentralBG_Cortical_feedback.lesion = 1.0
VentralBG_Cortical_feedback.learning=0.0 #what is this?

# Thal for WM/ventral loop, previously called Thal_vent
VentralBG_Thal = Population(name="VentralBG_Thal", geometry=num_objectives, neuron=LinearNeuron)
VentralBG_Thal.tau = 10.0
VentralBG_Thal.noise = 0.025 / 2.0
VentralBG_Thal.baseline = 1.2

# Reward cells of shell
VTA = Population(name='VTA',geometry=1,neuron=DopamineNeuron)
VTA.tau = 10.0
VTA.baseline = baseline_dopa_vent
VTA.factor_inh = 5.0

# reward independent of the stimulus in the hippocampus
# hippocampus activity
# previously called HippoRPELayer
HippoActivity= Population(name='HippoActivity',geometry=24,neuron=LinearNeuron)

#reward predction error
RPE = Population(name='RPE',geometry=(4,num_stim),neuron=LinearNeuron)
RPE.tau = 10.0
RPE.noise = 0.00
RPE.baseline = 0.0
RPE.lesion = 1.0

#This cells are active when one of the rewards is received -> ventral loop encodes task reward
#previsously called PPTN
VentralBG_PPTN = Population(name="VentralBG_PPTN", geometry=1, neuron=InputNeuron)
VentralBG_PPTN.tau = 1.0

# ~~~~~~~~~~~~ WORKING MEMORY ~~~~~~~~~~~~
#previously called StrD1_core
#shared with dorsomedial loop
WM_StrD1_core = Population(name='WM_StrD1_core', geometry=num_objectives, neuron=LinearNeuron)
WM_StrD1_core.tau = 10.0
WM_StrD1_core.noise = 0.0
WM_StrD1_core.baseline = 0.0
WM_StrD1_core.lesion = 1.0

#previously called SNr_core
WM_SNr_core = Population(name='WM_SNr_core', geometry=num_objectives, neuron=LinearNeuron)
WM_SNr_core.tau = 10.0
WM_SNr_core.noise = 0.3 
WM_SNr_core.baseline = 1.0#0.75 #1.5


# SYNAPSES
# shell loop 
ITStrD1_shell = []
ITStrD2_shell = []
StrD1VP_dir = []
StrD2VP_indir = []
ITSTN_shell = []
STNVP_dir = []
StrD1StrD1_shell = []
STNSTN_shell = []
StrD2StrD2_shell = []
# ITInter = []
# InterStrD1_shell = []
# InterStrD2_shell = []

#SYNAPSES
# did initially use the script of Scholl (2021), 
# where these populations were initialised this way, 
# basically implements all-to-all connections -> dont need this for loop
# in the end I left it this way

DAPostCovarianceNoThreshold = Synapse(
    parameters="""
        tau=1000.0
        tau_alpha=10.0
        regularization_threshold=1.0
        baseline_dopa = 0.1
        K_burst = 1.0
        K_dip = 0.4
        DA_type = 1
        threshold_pre=0.0
        threshold_post=0.0
    """,
    equations="""
        tau_alpha*dalpha/dt  + alpha = pos(post.mp - regularization_threshold)
        dopa_sum = 2.0*(post.sum(dopa) - baseline_dopa)
        trace = pos(post.r -  mean(post.r) - threshold_post) * (pre.r - mean(pre.r) - threshold_pre)
	    condition_0 = if (trace>0.0) and (w >0.0): 1 else: 0
        dopa_mod = if (DA_type*dopa_sum>0): DA_type*K_burst*dopa_sum else: condition_0*DA_type*K_dip*dopa_sum
        delta = (dopa_mod* trace - alpha*pos(post.r - mean(post.r) - threshold_post)*pos(post.r - mean(post.r) - threshold_post))
        tau*dw/dt = delta : min=0
    """
)

#Inhibitory synapses SNr -> SNr and STRD2 -> GPe
DAPreCovariance_inhibitory = Synapse(
    parameters="""
    tau=1000.0
    tau_alpha=10.0
    regularization_threshold=1.0
    baseline_dopa = 0.1
    K_burst = 1.0
    K_dip = 0.4
    DA_type= 1
    threshold_pre=0.0
    threshold_post=0.0
    negterm = 1
    """,
    equations="""
        tau_alpha*dalpha/dt = pos( -post.mp - regularization_threshold) - alpha
        dopa_sum = 2.0*(post.sum(dopa) - baseline_dopa)
        trace = pos(pre.r - mean(pre.r) - threshold_pre) * (mean(post.r) - post.r  - threshold_post)
        aux = if (trace>0): negterm else: 0
        dopa_mod = if (DA_type*dopa_sum>0): DA_type*K_burst*dopa_sum else: aux*DA_type*K_dip*dopa_sum
        trace2 = trace
        delta = dopa_mod * trace2 - alpha * pos(trace2)
        tau*dw/dt = delta : min=0
    """
)

#Excitatory synapses STN -> SNr
DAPreCovariance_excitatory = Synapse(
    parameters="""
    tau=1000.0
    tau_alpha=10.0
    regularization_threshold=1.0
    baseline_dopa = 0.1
    K_burst = 1.0
    K_dip = 0.4
    DA_type= 1
    threshold_pre=0.0
    threshold_post=0.0
    """,
    equations = """
        tau_alpha*dalpha/dt  = pos( post.mp - regularization_threshold) - alpha
        dopa_sum = 2.0*(post.sum(dopa) - baseline_dopa)
        trace = pos(pre.r - mean(pre.r) - threshold_pre) * (post.r - mean(post.r) - threshold_post)
        aux = if (trace<0.0): 1 else: 0
        dopa_mod = if (dopa_sum>0): K_burst * dopa_sum else: K_dip * dopa_sum * aux
        delta = dopa_mod * trace - alpha * pos(trace)
        tau*dw/dt = delta : min=0
    """
)

ReversedSynapse = Synapse(
    parameters="""
        reversal = 0.3
    """,
    psp="""
        w*pos(reversal-pre.r)
    """
)

DAPrediction = Synapse(
    parameters="""
        tau = 100000.0
        baseline_dopa = 0.1
   """,
   equations="""
       aux = if (post.sum(exc) > 0.001): 1.0 else: 2.5
       delta = aux*(post.r-baseline_dopa)*pos(pre.r-mean(pre.r))
       tau*dw/dt = delta : min=0
   """
)

# PROJECTIONS FOR VENTRAL LOOP
for i in range(0,num_stim):
    
    #Input from the stimulus representing cells to the striatum, main plastic connections to the loop.
    #The input goes to the 2 striatal populations
    #create a projection for every column of neurons from hipocampus to NAccD1
    #the store it in an array
    ITStrD1_shell.append(Projection(pre=Hippocampus,post=NAccD1_shell[:,i],target='exc',synapse=DAPostCovarianceNoThreshold))
    ITStrD1_shell[i].connect_all_to_all(weights = Normal(0.1/(num_stim/2),0.02)) # perhaps also scale by numer of goals #0.02
    ITStrD1_shell[i].tau = 200 #100
    ITStrD1_shell[i].regularization_threshold = 1.0
    ITStrD1_shell[i].tau_alpha = 5.0
    ITStrD1_shell[i].baseline_dopa = baseline_dopa_vent
    ITStrD1_shell[i].K_dip = 0.05
    ITStrD1_shell[i].K_burst = 1.0
    ITStrD1_shell[i].DA_type = 1
    ITStrD1_shell[i].threshold_pre = 0.2
    ITStrD1_shell[i].threshold_post = 0.0

    #same for the indirect pathway
    ITStrD2_shell.append(Projection(pre=Hippocampus,post=NAccD2_shell[:,i],target='exc',synapse=DAPostCovarianceNoThreshold))
    ITStrD2_shell[i].connect_all_to_all(weights = Normal(0.01/(num_stim/2),0.01)) #0.005
    ITStrD2_shell[i].tau = 10.0
    ITStrD2_shell[i].regularization_threshold = 1.5
    ITStrD2_shell[i].tau_alpha = 15.0
    ITStrD2_shell[i].baseline_dopa = baseline_dopa_vent
    ITStrD2_shell[i].K_dip = 0.2
    ITStrD2_shell[i].K_burst = 1.0
    ITStrD2_shell[i].DA_type = -1
    ITStrD2_shell[i].threshold_pre = 0.2
    ITStrD2_shell[i].threshold_post = 0.005

    #same logic for the connection between hippocampus and STN
    ITSTN_shell.append(Projection(pre=Hippocampus, post=VentralBG_STN[:,i], target='exc',synapse=DAPostCovarianceNoThreshold))
    ITSTN_shell[i].connect_all_to_all(weights = Uniform(0.0,0.001))
    ITSTN_shell[i].tau = 1500.0
    ITSTN_shell[i].regularization_threshold = 1.0
    ITSTN_shell[i].tau_alpha = 15.0
    ITSTN_shell[i].baseline_dopa = baseline_dopa_vent
    ITSTN_shell[i].K_dip = 0.0 #0.4
    ITSTN_shell[i].K_burst = 1.0
    ITSTN_shell[i].DA_type = 1
    ITSTN_shell[i].threshold_pre = 1.5
    ITSTN_shell[i].threshold_post = 0.0

    #indirect pathway
    StrD2VP_indir.append(Projection(pre=NAccD2_shell[:,i],post=VentralBG_VP_indir,target='inh',synapse=DAPreCovariance_inhibitory))
    StrD2VP_indir[i].connect_all_to_all(weights=(0.01)) # was 0.01
    StrD2VP_indir[i].tau = 1000 
    StrD2VP_indir[i].regularization_threshold = 1.5 
    StrD2VP_indir[i].tau_alpha = 20.0 
    StrD2VP_indir[i].baseline_dopa = baseline_dopa_vent
    StrD2VP_indir[i].K_dip = 0.1
    StrD2VP_indir[i].K_burst = 1.0
    StrD2VP_indir[i].threshold_post = 0.0
    StrD2VP_indir[i].threshold_pre = 0.25
    StrD2VP_indir[i].DA_type = -1

    #Hyperdirect
    STNVP_dir.append(Projection(pre=VentralBG_STN[:,i],post=VentralBG_VP_dir,target='exc',synapse=DAPreCovariance_excitatory))
    #STNSNr_caudate[i].connect_all_to_all(weights=Uniform(0.0012,0.0014))
    STNVP_dir[i].connect_all_to_all(weights = Uniform(0.01,0.001))
    STNVP_dir[i].tau = 9000
    STNVP_dir[i].regularization_threshold = 3.5 # increased this from 1.5, because otherwise all weights went to 0
    STNVP_dir[i].tau_alpha = 1.0
    STNVP_dir[i].baseline_dopa = baseline_dopa_vent
    STNVP_dir[i].K_dip = 0.4
    STNVP_dir[i].K_burst = 1.0
    STNVP_dir[i].threshold_post = -0.15 #-0.5 #-0.15
    STNVP_dir[i].DA_type = 1

    # local inhibitory connections
    STNSTN_shell.append(Projection(pre=VentralBG_STN[:,i],post=VentralBG_STN[:,i],target='inh'))
    STNSTN_shell[i].connect_all_to_all(weights= weight_stn_inh)

    StrD2StrD2_shell.append(Projection(pre=NAccD2_shell[:,i],post=NAccD2_shell[:,i],target='inh'))
    StrD2StrD2_shell[i].connect_all_to_all(weights= weight_inh_sd2)

#idk why this is defined as an array and then changed into a projection
StrD1StrD1_shell=Projection(pre=NAccD1_shell,post=NAccD1_shell,target='inh')
StrD1StrD1_shell.connect_all_to_all(weights = weight_local_inh) 

# following the direct pathway
StrD1VP_dir=Projection(pre=NAccD1_shell,post=VentralBG_VP_dir,target='inh',synapse=DAPreCovariance_inhibitory)
StrD1VP_dir.connect_all_to_all(weights=Normal(0.2,0.01)) # scale by numer of stimuli #var 0.01
StrD1VP_dir.tau = 2000 #700 #550
StrD1VP_dir.regularization_threshold = 3.5 #2 #3.5 #1.5
StrD1VP_dir.tau_alpha = 2.5 # 20.0
StrD1VP_dir.baseline_dopa = baseline_dopa_vent
StrD1VP_dir.K_dip = 0.8 #0.9
StrD1VP_dir.K_burst = 1.2
StrD1VP_dir.threshold_post = 0.4 #0.1 #0.3
StrD1VP_dir.threshold_pre = 0.1 # 0.15
StrD1VP_dir.DA_type=1
StrD1VP_dir.negterm = 5.0

# also plastic connections for the Cells encoding the reward prediction
HippoRPE = Projection(pre=Hippocampus,post=HippoActivity[:8],target='exc')
HippoRPE.connect_all_to_all(weights=1.0/8.0)
HippoRPELayerRPE = Projection(pre=HippoActivity,post=RPE,target='exc',synapse=DAPostCovarianceNoThreshold)
HippoRPELayerRPE.connect_all_to_all(weights = Normal(0.1/(num_stim/2),0.02)) # perhaps also scale by numer of goals #0.02
HippoRPELayerRPE.tau = 200 #100
HippoRPELayerRPE.regularization_threshold = 1.0
HippoRPELayerRPE.tau_alpha = 5.0
HippoRPELayerRPE.baseline_dopa = baseline_dopa_vent
HippoRPELayerRPE.K_dip = 0.05
HippoRPELayerRPE.K_burst = 1.0
HippoRPELayerRPE.DA_type = 1
HippoRPELayerRPE.threshold_pre = 0.2
HippoRPELayerRPE.threshold_post = 0.00

# local effects using fixed connections
RPERPE=Projection(pre=RPE,post=RPE,target='inh')
RPERPE.connect_all_to_all(weights = weight_local_inh)

StrThalStrThal_shell = Projection(pre=VentralBG_Cortical_feedback,post=VentralBG_Cortical_feedback,target='inh')
StrThalStrThal_shell.connect_all_to_all(weights=weight_inh_thal) #0.5

VPVP_shell = Projection(pre=VentralBG_VP_dir,post=VentralBG_VP_dir,target='exc',synapse=ReversedSynapse)
VPVP_shell.connect_all_to_all(weights=0.7)#0.7
VPVP_shell.reversal = 0.3

ThalThal = Projection(pre=VentralBG_Thal,post=VentralBG_Thal,target='inh')
ThalThal.connect_all_to_all(weights=0.1) # was 1.1 # perhaps needs to be lower

VP_indirVP_dir = Projection(pre=VentralBG_VP_indir,post=VentralBG_VP_dir,target='inh')
VP_indirVP_dir.connect_one_to_one(weights=0.5) 

StrThalVP_indir = Projection(pre=VentralBG_Cortical_feedback,post=VentralBG_VP_indir,target='inh')
StrThalVP_indir.connect_one_to_one(weights=0.3)

StrThalVP_dir = Projection(pre=VentralBG_Cortical_feedback,post=VentralBG_VP_dir,target='inh')
StrThalVP_dir.connect_one_to_one(weights=1.1)

VPThal = Projection(pre=VentralBG_VP_dir,post=VentralBG_Thal,target='inh')
VPThal.connect_one_to_one(weights=0.75) # 1.5

ObjObj = Projection(pre=mPFC,post=mPFC,target='inh')
ObjObj.connect_all_to_all(weights=0.7) #0.75

VTAStrD1_shell = []
VTAStrD2_shell = []
VTASTN_shell= []
StrD1VTA_core = []

# get the Feedback for the Reward Predction from a seperate group of cells with an constant input instead of an everchanging one
RPEVTA=Projection(pre=RPE,post=VTA,target='inh',synapse=DAPrediction)
RPEVTA.connect_all_to_all(weights=0.0)
RPEVTA.tau = 4500#3000
RPEVTA.baseline_dopa=baseline_dopa_vent

# dopamine output encoded in VTA activity
VTAStrD1_shell.append(Projection(pre=VTA,post=NAccD1_shell,target='dopa'))
VTAStrD1_shell[-1].connect_all_to_all(weights=1.0) #weights=1.0/(num_stim/3)

VTAStrD2_shell.append(Projection(pre=VTA,post=NAccD2_shell,target='dopa'))
VTAStrD2_shell[-1].connect_all_to_all(weights=1.0)

VTASTN_shell.append(Projection(pre=VTA,post=VentralBG_STN,target='dopa'))
VTASTN_shell[-1].connect_all_to_all(weights=1.0)

VTARPE = Projection(pre=VTA,post=RPE,target='dopa')
VTARPE.connect_all_to_all(weights=1.0)    

VTAVP_dir = Projection(pre=VTA,post=VentralBG_VP_dir,target='dopa')
VTAVP_dir.connect_all_to_all(weights=1.0)

VTAVP_indir = Projection(pre=VTA,post=VentralBG_VP_indir,target='dopa')
VTAVP_indir.connect_all_to_all(weights=1.0)

# task reward
PPTNVTA = Projection(pre=VentralBG_PPTN,post=VTA,target='exc')
PPTNVTA.connect_one_to_one(weights=1.0)

## handcrafted working memory
ThalObj = Projection(pre=VentralBG_Thal,post=mPFC,target='exc')
ThalObj.connect_one_to_one(weights=1.0) #3.0

ObjStrD1_core = Projection(pre=mPFC,post=WM_StrD1_core,target='exc')
ObjStrD1_core.connect_one_to_one(weights=2.0)    

StrD1SNr_core = Projection(pre=WM_StrD1_core,post=WM_SNr_core,target='inh')
StrD1SNr_core.connect_one_to_one(weights=0.75)

SNrThal_core = Projection(pre=WM_SNr_core,post=VentralBG_Thal,target='inh')
SNrThal_core.connect_one_to_one(weights=1.2)

ObjStrThal_shell= Projection(pre=mPFC,post=VentralBG_Cortical_feedback,target='exc')
ObjStrThal_shell.connect_one_to_one(weights=1.2)

ObjObj = Projection(pre=mPFC,post=mPFC,target='inh')
ObjObj.connect_all_to_all(weights=0.8) 

compile()

def get_first_action(sequence):
    if sequence < 4:
        return '1'
    elif sequence >=4:
        return '2'
    else:
        return None

def diffuse_prob(prob):
    """Diffuses a probability between 0.25 and 0.75"""
    prob += random.gauss(0, 0.025)
    if prob < 0.25:
        prob = 0.5 - prob
    elif prob > 0.75:
        prob = 1.5 - prob
    assert prob >= 0.25 and prob <= 0.75
    return prob

def objective(trial):
    tau_NAccD1_shell = trial['tau_NAccD1_shell']
    noise_NAccD1_shell = trial['noise_NAccD1_shell']
    baseline_NAccD1_shell = trial['baseline_NAccD1_shell']
    lesion_NAccD1_shell = trial['lesion_NAccD1_shell']
    tau_trace_NAccD1_shell = trial['tau_trace_NAccD1_shell']

    tau_VentralBG_VP_dir = trial['tau_VentralBG_VP_dir']
    noise_VentralBG_VP_dir = trial['noise_VentralBG_VP_dir']
    baseline_VentralBG_VP_dir = trial['baseline_VentralBG_VP_dir']
    tau_trace_VentralBG_VP_dir = trial['tau_trace_VentralBG_VP_dir']

    tau_VentralBG_Thal = trial['tau_VentralBG_Thal']
    noise_VentralBG_Thal = trial['noise_VentralBG_Thal']
    baseline_VentralBG_Thal = trial['baseline_VentralBG_Thal']
    
    #hippo to strd1
    tau_ITStrD1_shell = trial['tau_ITStrD1_shell']
    regularization_threshold_ITStrD1_shell = trial['regularization_threshold_ITStrD1_shell']
    tau_alpha_ITStrD1_shell = trial['tau_alpha_ITStrD1_shell']
    baseline_dopa_ITStrD1_shell = trial['baseline_dopa_ITStrD1_shell']
    k_dip_ITStrD1_shell = trial['k_dip_ITStrD1_shell']
    k_burst_ITStrD1_shell = trial['k_burst_ITStrD1_shell']
    threshold_pre_ITStrD1_shell = trial['threshold_pre_ITStrD1_shell']
    threshold_post_ITStrD1_shell = trial['threshold_post_ITStrD1_shell']

    #strd1 to vpdir
    tau_StrD1VP_dir = trial['tau_StrD1VP_dir']
    regularization_threshold_StrD1VP_dir = trial['regularization_threshold_StrD1VP_dir']
    tau_alpha_StrD1VP_dir = trial['tau_alpha_StrD1VP_dir']
    baseline_dopa_StrD1VP_dir = trial['baseline_dopa_StrD1VP_dir']
    k_dip_StrD1VP_dir = trial['k_dip_StrD1VP_dir']
    k_burst_StrD1VP_dir = trial['k_burst_StrD1VP_dir']
    threshold_pre_StrD1VP_dir = trial['threshold_pre_StrD1VP_dir']
    threshold_post_StrD1VP_dir = trial['threshold_post_StrD1VP_dir']
    
    #vpdir to thal
    w_VPThal = trial['w_VPThal']
    w_ObjObj = trial['w_ObjObj']

    w_StrD1StrD1_shell = trial['w_StrD1StrD1_shell']
    w_VPVP_shell = trial['w_VPVP_shell']
    
    w_ThalThal = trial['w_ThalThal']
    w_StrThalVP_dir = trial['w_StrThalVP_dir']

    tau_HippoRPELayerRPE = trial['tau_HippoRPELayerRPE']
    tau_RPEVTA = trial['tau_RPEVTA']

    NAccD1_shell.tau = tau_NAccD1_shell
    NAccD1_shell.noise = noise_NAccD1_shell
    NAccD1_shell.baseline = baseline_NAccD1_shell
    NAccD1_shell.lesion = lesion_NAccD1_shell
    NAccD1_shell.tau_trace = tau_trace_NAccD1_shell

    VentralBG_VP_dir.tau = tau_VentralBG_VP_dir
    VentralBG_VP_dir.noise = noise_VentralBG_VP_dir
    VentralBG_VP_dir.baseline = baseline_VentralBG_VP_dir
    VentralBG_VP_dir.tau_trace = tau_trace_VentralBG_VP_dir

    VentralBG_Thal.tau = tau_VentralBG_Thal
    VentralBG_Thal.noise = noise_VentralBG_Thal
    VentralBG_Thal.baseline = baseline_VentralBG_Thal
    
    for pop in ITStrD1_shell:
        pop.tau = tau_ITStrD1_shell
        pop.regularization_threshold = regularization_threshold_ITStrD1_shell
        pop.tau_alpha = tau_alpha_ITStrD1_shell
        pop.baseline_dopa = baseline_dopa_ITStrD1_shell
        pop.K_dip = k_dip_ITStrD1_shell
        pop.K_burst = k_burst_ITStrD1_shell
        pop.threshold_pre = threshold_pre_ITStrD1_shell
        pop.threshold_post = threshold_post_ITStrD1_shell

    StrD1StrD1_shell.w = w_StrD1StrD1_shell

    StrD1VP_dir.tau = tau_StrD1VP_dir
    StrD1VP_dir.regularization_threshold = regularization_threshold_StrD1VP_dir
    StrD1VP_dir.tau_alpha = tau_alpha_StrD1VP_dir
    StrD1VP_dir.baseline_dopa = baseline_dopa_StrD1VP_dir
    StrD1VP_dir.K_dip = k_dip_StrD1VP_dir
    StrD1VP_dir.K_burst = k_burst_StrD1VP_dir
    StrD1VP_dir.threshold_pre = threshold_pre_StrD1VP_dir
    StrD1VP_dir.threshold_post = threshold_post_StrD1VP_dir

    VPVP_shell.w = w_VPVP_shell
    ThalThal.w = w_ThalThal
    StrThalVP_dir.w = w_StrThalVP_dir
    VPThal.w = w_VPThal
    ObjObj.w = w_ObjObj

    RPEVTA.tau = tau_RPEVTA
    HippoRPELayerRPE.tau = tau_HippoRPELayerRPE


    sim_no = 50
    simulations_probs=[]
    enable_learning()
    for sim in range(sim_no):
        # initialize some arrays for the results
        reward_of_actions=[]
        dopamine_activities=[]
        achieved_desired_sequence=[]

        # init the task transition probability 
        transition_prob=0.7
        num_trials = 300
        eps=0.000001
        stimulis=list(range(8))
        stim_to_action={0:"(1,A)",1:"(1,B)",2:"(1,C)",3:"(1,D)",4:"(2,A)",5:"(2,B)",6:"(2,C)",7:"(2,D)"}

        #iterations done by one person
        acc_reward = 0
        rewarded=[]
        selected_objectives=[]
        achieved_sequences=[]
        reward_of_actions=[]
        realizable_stimulus=[]
        common_transition=[]     

        Hippocampus.noise = 0.0 #0.001#0.001     

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

            #intertrial
            simulate(2000)

            VentralBG_Thal.baseline=1.2
            # reset finished

            simulate(200)   

            # show the stimuli in random order
            random.shuffle(stimulis)  

            # show each stimuli for 300 ms
            # stimuli is represented by an integer eg. (3,4,7,0,1,5,2,6)
            # check with sim_to_action
            for stim in stimulis:   
                Hippocampus[:,stim//4].baseline=1.0 
                Hippocampus[:,2+stim%4].baseline=1.0
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

            # short pause to check if WM retains sequence / has decided on one
            simulate(200)

            #objective action depending on mPFC, these represents same transition as sim_to_action
            selected_objective=random.choices(list(range(8)),weights=(mPFC.r+eps),k=1)[0]
            selected_objectives.append(selected_objective)
            #value [0,1] -> 1,2
            first_action = selected_objective//4
            #value [0,1,2,3] -> A,B,C,D
            second_action = selected_objective%4

            #action 1
            if first_action==0:
                #common transition
                if random.random()<transition_prob:
                    second_panel=1
                    common_transition.append(1)
                #rare transition
                else:
                    second_panel=2
                    common_transition.append(0)
            #action 2
            if first_action==1:
                #common transition
                if random.random()<transition_prob:
                    second_panel=2
                    common_transition.append(1)
                #rare transition
                else:
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

            #some time for second decision -> can be shorter if one wants that
            simulate(1000)

            #if not realizable, choose a random second action
            if not realizable:
                second_action = random.choices([0,1],k=1)[0]
                #second action is in the range [0,1,2,3] -> [A,B,C,D]
                if second_panel==2:
                    second_action+=2


            #calculate the sequence selected so it's in the range [0,7]
            #eg. if sequence is 5 -> (2,B) this means that first and second action are 1 and 1 respectively
            #that gives a total of 4*1*+1 = 5, you can calculate the others to check
            achieved_sequence=4*first_action+second_action
            achieved_sequences.append(achieved_sequence)

            ## LEARNING PART
            #update columns for the selected sequence
            mPFC[achieved_sequence].baseline=2.0
            Hippocampus[:,first_action].baseline=0.5 
            Hippocampus[:,2+second_action].baseline=0.5


            # first short step with higer intergration
            simulate(100)

            # now reduce the feedback strenght
            mPFC[achieved_sequence].baseline=0.2
            VentralBG_Cortical_feedback.learning=1

            #again, short time of cortical integration
            simulate(300)

            # task reward
            reward_random=random.random()
            #get last item form array
            #this array is initialized before, check if the second action has a reward
            #rewarded
            rew = 0
            if reward_random < reward_of_actions[second_action]:
                VentralBG_PPTN.baseline=1.0
                rewarded.append(1)
                rew = 1
            #non-rewarded
            else:
                rewarded.append(0)

            ## if we want to delay learning in the ventral loop for the first 100 trials
            #if trial >= 100: 
            VTA.firing=1  

            #actual learning period, outcome
            simulate(100)

            VentralBG_PPTN.baseline=0.0
            VTA.firing=0
            VentralBG_Cortical_feedback.learning=0    

            dopamine_activities.append(VTA.r)
            #this stores if the sequence proposed by the mPFC is actually selected by the other loops
            #this is not sored, you could
            achieved_desired_sequence.append(selected_objective==achieved_sequence)
            
            # change the rewards
            for i in range(4):
                reward_of_actions[i]=diffuse_prob(reward_of_actions[i])
            
        data_dict = {
            'achieved_sequences': achieved_sequences,
            'reward': rewarded,
            'transition': common_transition,
        }

        data = pd.DataFrame(data_dict)    
        data['first_action'] = data['achieved_sequences'].apply(get_first_action)
        data['next_first_action'] = data['first_action'].shift(-1)
        data = data.fillna(0.0)

        graph_data = data[['first_action', 'next_first_action', 'reward', 'transition']].copy()
        #same first action, rewarded and common
        #same first action, rewarded and common
        graph_data['stay_common_rewarded'] = np.where(
            (graph_data['first_action'] == graph_data['next_first_action']) &
            (graph_data['reward']==1.0) &
            (graph_data['transition']==1.0),
            True, False)
        graph_data['common_rewarded'] = np.where(
            (graph_data['reward']==1.0) &
            (graph_data['transition']==1.0),
            True, False)

        #same first action, rewarded and rare
        graph_data['stay_rare_rewarded'] = np.where(
            (graph_data['first_action'] == graph_data['next_first_action']) &
            (graph_data['reward']==1.0) &
            (graph_data['transition']==0.0)
            , True, False)
        graph_data['rare_rewarded'] = np.where(
            (graph_data['reward']==1.0) &
            (graph_data['transition']==0.0),
            True, False)

        graph_data['stay_common_unrewarded'] = np.where(
            (graph_data['first_action'] == graph_data['next_first_action']) &
            (graph_data['reward']==0.0) &
            (graph_data['transition']==1.0)
            , True, False)
        graph_data['common_unrewarded'] = np.where(
            (graph_data['reward']==0.0) &
            (graph_data['transition']==1.0),
            True, False)

        graph_data['stay_rare_unrewarded'] = np.where(
            (graph_data['first_action'] == graph_data['next_first_action']) &
            (graph_data['reward']==0.0) &
            (graph_data['transition']==0.0)
            , True, False)
        graph_data['rare_unrewarded'] = np.where(
            (graph_data['reward']==0.0) &
            (graph_data['transition']==0.0),
            True, False)
        
        probs = []
        probs.append(graph_data['stay_common_rewarded'].sum() / graph_data['common_rewarded'].sum())
        probs.append(graph_data['stay_rare_rewarded'].sum() / graph_data['rare_rewarded'].sum())
        probs.append(graph_data['stay_common_unrewarded'].sum() / graph_data['common_unrewarded'].sum())
        probs.append(graph_data['stay_rare_unrewarded'].sum() / graph_data['rare_unrewarded'].sum())

        simulations_probs.append(probs)


    avg_probs = np.average(simulations_probs, axis=0)
    print("avg probabilities", avg_probs)
    true_target = [0.85, 0.76, 0.6, 0.72]
    loss = mean_squared_error(true_target, avg_probs)

    return loss

# Assuming you have a completed Optuna study
study = optuna.create_study(
    study_name='bg_bigger_dir_only',
    storage='mysql://optuna:123456@localhost:3306/optuna',
    load_if_exists=True
)

best_trial = study.best_trial
best_params = best_trial.params

# Re-run with fixed random seeds
fixed_random_seed = 42  # Choose a specific seed

# Re-execute your objective function with best_params and fixed_random_seed
result = objective(best_params)

# Evaluate the re-run result
print(result)