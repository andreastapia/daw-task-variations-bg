from ANNarchy import *
import pylab as plt
import random
import sys
import scipy.spatial.distance

#General networks parameters
baseline_dopa_vent = 0.15
baseline_dopa_caud=0.1
gen_tau_trace=100
num_stim = 6
num_objectives=8

#neurons_per_visual = 2
neurons_per_goal = 3

#Neuron models
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
# non-linear effect of input on inhibition
TAInterneuron = Neuron(
    parameters="""
    tau = 10.0
    baseline = 0.1
    noise = 0.01
    """,
    equations = """
    tau*dmp/dt + mp = baseline + power(sum(exc),3) + noise*Uniform(-1.0,1.0)
    r = if (mp>0.0): mp else: 0.0
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

#if (firing>0): (ex_in)*(pos(1.0-baseline-s_inh) + baseline) + (1-ex_in)*(-factor_inh*sum(inh)+baseline)  else: baseline

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

###################################################################################################################################################
###################################################################################################################################################

#Synapse models
ReversedSynapse = Synapse(
    parameters="""
        reversal = 0.3
    """,
    psp="""
        w*pos(reversal-pre.r)
    """
)

#DA_typ = 1  ==> D1 type  DA_typ = -1 ==> D2 type
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
        trace = pos(pre.r - mean(pre.r) - threshold_pre) * (post.trace - mean(post.trace) - threshold_post)
        aux = if (trace<0.0): 1 else: 0
        dopa_mod = if (dopa_sum>0): K_burst * dopa_sum else: K_dip * dopa_sum * aux
        delta = dopa_mod * trace - alpha * pos(trace)
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

###################################################################################################################################################
###################################################################################################################################################

# Neurons for the ventral loop
#mPFC
#1
Objectives = Population(name="Objectives", geometry=num_objectives, neuron=LinearNeuron)
Objectives.tau = 40.0 # 40
Objectives.noise = 0.01

#2
# Hippocampus is implemented as Input neurons cycling through the different sequences 
Input_neurons = Population(name='Input',geometry=(4,num_stim),neuron=InputNeuron)

#Hippocampus = Population(name='Hippocampus',geometry=num_stim,neuron=LinearNeuron)

# Nacc Shell 
# D1-type striatum neurons
#3
StrD1_shell = Population(name="StrD1_shell", geometry=(4, num_stim),neuron = LinearNeuron_trace)
StrD1_shell.tau = 10.0
StrD1_shell.noise = 0.3
StrD1_shell.baseline = 0.0
StrD1_shell.lesion = 1.0
StrD1_shell.tau_trace = gen_tau_trace

# shell indirect pathway 
# similar to the direct
#4
StrD2_shell = Population(name="StrD2_shell", geometry = (4, num_stim), neuron=LinearNeuron_trace)
StrD2_shell.tau = 10.0
StrD2_shell.noise = 0.01
StrD2_shell.baseline = 0.0
StrD2_shell.lesion = 1.0
StrD1_shell.tau_trace = gen_tau_trace


# shell VP (direct)
#5
VP_dir = Population(name="VP_dir", geometry = num_objectives, neuron=LinearNeuron_trace)
VP_dir.tau = 10.0
VP_dir.noise = 0.3 #0.3
VP_dir.baseline = 1.5 #0.75 #1.5
VP_dir.tau_trace = gen_tau_trace

# shell VP (indirect)
#6
VP_indir = Population(name="VP_indir", geometry = num_objectives, neuron=LinearNeuron_trace)
VP_indir.tau = 10.0
VP_indir.noise = 0.05
VP_indir.baseline = 1.0
VP_dir.tau_trace = gen_tau_trace


# STN
#7
STN_shell = Population(name="STN_shell", geometry = (4, num_stim), neuron=LinearNeuron_trace)
STN_shell.tau = 10.0
STN_shell.noise = 0.01
STN_shell.baseline = 0.0
StrD1_shell.tau_trace = gen_tau_trace

# Feedback from Cortex/Thal to GPi
# enables hebbian learning rule by excitation of the VP neurons
#8
StrThal_shell = Population(name="StrThal_shell", geometry = num_objectives, neuron=LinearNeuron_learning)
StrThal_shell.tau = 10.0
StrThal_shell.noise = 0.01
StrThal_shell.baseline = 0.4
StrThal_shell.lesion = 1.0
StrThal_learning=0.0

# Thal
#9
Thal_vent = Population(name="Thal_vent", geometry=num_objectives, neuron=LinearNeuron)
Thal_vent.tau = 10.0
Thal_vent.noise = 0.025 / 2.0
Thal_vent.baseline = 1.2

# Reward cells of shell
#10
VTA = Population(name='VTA',geometry=1,neuron=DopamineNeuron)
VTA.baseline = baseline_dopa_vent
VTA.factor_inh = 5.0

# reward independent of the stimulus in the hippocampus
#11
HippoRPELayer = Population(name='HippoRPELayer',geometry=24,neuron=LinearNeuron)
#12
RPE = Population(name='RPE',geometry=(4,num_stim),neuron=LinearNeuron)
RPE.tau = 10.0
RPE.noise = 0.00
RPE.baseline = 0.0
RPE.lesion = 1.0

#This cells are active when one of the rewards is received -> ventral loop encodes task reward
#13
PPTN = Population(name="PPTN", geometry=1, neuron=InputNeuron)
PPTN.tau = 1.0

## Working memory
#14
StrD1_core = Population(name='StrD1_core', geometry=num_objectives, neuron=LinearNeuron)
StrD1_core.tau = 10.0
StrD1_core.noise = 0.0
StrD1_core.baseline = 0.0
StrD1_core.lesion = 1.0

#15
SNr_core = Population(name='SNr_core', geometry=num_objectives, neuron=LinearNeuron)
SNr_core.tau = 10.0
SNr_core.noise = 0.3 
SNr_core.baseline = 1.0#0.75 #1.5


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
ITInter = []
InterStrD1_shell = []
InterStrD2_shell = []

# caudate
weight_local_inh = 0.8 *0.2 * 0.166 #old model (Scholl) with 6 distinct population -> 0.8*0.2, here only one -> scale it by 1/6
weight_stn_inh = 0.3 
weight_inh_sd2 = 0.5 *0.2
weight_inh_thal = 0.5 *0.2
modul = 0.01

#SYNAPSES
# did initially use the script of Scholl (2021), where these populations were initialised this way, basically implements all-to-all connections -> dont need this for loop
# in the end I left it this way
for i in range(0,num_stim):
    #Input from the stimulus representing cells to the striatum, main plastic connections to the loop.
    #The input goes to the 2 striatal populations
    ITStrD1_shell.append(Projection(pre=Input_neurons,post=StrD1_shell[:,i],target='exc',synapse=DAPostCovarianceNoThreshold))
    ITStrD1_shell[i].connect_all_to_all(weights = Normal(0.1/(num_stim/2),0.02)) # perhaps also scale by numer of goals #0.02
    ITStrD1_shell[i].tau = 200 #100                          #tau w
    ITStrD1_shell[i].regularization_threshold = 1.0          #m max
    ITStrD1_shell[i].tau_alpha = 5.0                         #tau a
    ITStrD1_shell[i].baseline_dopa = baseline_dopa_vent      
    ITStrD1_shell[i].K_dip = 0.05                            #Kd
    ITStrD1_shell[i].K_burst = 1.0                           #Kb
    ITStrD1_shell[i].DA_type = 1                             #Td
    ITStrD1_shell[i].threshold_pre = 0.2                     #gamma pre
    ITStrD1_shell[i].threshold_post = 0.0                    #gamma post

    #same for the indirect pathway
    ITStrD2_shell.append(Projection(pre=Input_neurons,post=StrD2_shell[:,i],target='exc',synapse=DAPostCovarianceNoThreshold))
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

    ITSTN_shell.append(Projection(pre=Input_neurons, post=STN_shell[:,i], target='exc',synapse=DAPostCovarianceNoThreshold))
    ITSTN_shell[i].connect_all_to_all(weights = Uniform(0.0,0.001))
    ITSTN_shell[i].tau = 1500.0
    ITSTN_shell[i].regularization_threshold = 1.0
    ITSTN_shell[i].tau_alpha = 15.0
    ITSTN_shell[i].baseline_dopa = baseline_dopa_vent
    ITSTN_shell[i].K_dip = 0.0 #0.4
    ITSTN_shell[i].K_burst = 1.0
    ITSTN_shell[i].DA_type = 1
    ITSTN_shell[i].threshold_pre = 0.15 #1.5 #0.15 #correction from paper
    ITSTN_shell[i].threshold_post = 0.0 #correction from paper

    #indirect pathway
    StrD2VP_indir.append(Projection(pre=StrD2_shell[:,i],post=VP_indir,target='inh',synapse=DAPreCovariance_inhibitory))
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
    STNVP_dir.append(Projection(pre=STN_shell[:,i],post=VP_dir,target='exc',synapse=DAPreCovariance_excitatory))
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
    STNSTN_shell.append(Projection(pre=STN_shell[:,i],post=STN_shell[:,i],target='inh'))
    STNSTN_shell[i].connect_all_to_all(weights= weight_stn_inh)

    StrD2StrD2_shell.append(Projection(pre=StrD2_shell[:,i],post=StrD2_shell[:,i],target='inh'))
    StrD2StrD2_shell[i].connect_all_to_all(weights= weight_inh_sd2)

StrD1StrD1_shell=Projection(pre=StrD1_shell,post=StrD1_shell,target='inh')
StrD1StrD1_shell.connect_all_to_all(weights = weight_local_inh) 



# following the direct pathway
StrD1VP_dir=Projection(pre=StrD1_shell,post=VP_dir,target='inh',synapse=DAPreCovariance_inhibitory)
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
HippoRPE = Projection(pre=Input_neurons,post=HippoRPELayer[:8],target='exc')
HippoRPE.connect_all_to_all(weights=1.0/8.0)
HippoRPELayerRPE = Projection(pre=HippoRPELayer,post=RPE,target='exc',synapse=DAPostCovarianceNoThreshold)
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

StrThalStrThal_shell = Projection(pre=StrThal_shell,post=StrThal_shell,target='inh')
StrThalStrThal_shell.connect_all_to_all(weights=weight_inh_thal) #0.5

VPVP_shell = Projection(pre=VP_dir,post=VP_dir,target='exc',synapse=ReversedSynapse)
VPVP_shell.connect_all_to_all(weights=0.7)#0.7
VPVP_shell.reversal = 0.3

ThalThal = Projection(pre=Thal_vent,post=Thal_vent,target='inh')
ThalThal.connect_all_to_all(weights=0.2)  # 0.1 correction from paper 0.2 # was 1.1 # perhaps needs to be lower

VP_indirVP_dir = Projection(pre=VP_indir,post=VP_dir,target='inh')
VP_indirVP_dir.connect_one_to_one(weights=0.5) 

StrThalVP_indir = Projection(pre=StrThal_shell,post=VP_indir,target='inh')
StrThalVP_indir.connect_one_to_one(weights=0.3)

StrThalVP_dir = Projection(pre=StrThal_shell,post=VP_dir,target='inh')
StrThalVP_dir.connect_one_to_one(weights=1.1)

VPThal = Projection(pre=VP_dir,post=Thal_vent,target='inh')
VPThal.connect_one_to_one(weights=0.75) # 1.5

ObjObj = Projection(pre=Objectives,post=Objectives,target='inh')
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
VTAStrD1_shell.append(Projection(pre=VTA,post=StrD1_shell,target='dopa'))
VTAStrD1_shell[-1].connect_all_to_all(weights=1.0) #weights=1.0/(num_stim/3)

VTAStrD2_shell.append(Projection(pre=VTA,post=StrD2_shell,target='dopa'))
VTAStrD2_shell[-1].connect_all_to_all(weights=1.0)

VTASTN_shell.append(Projection(pre=VTA,post=STN_shell,target='dopa'))
VTASTN_shell[-1].connect_all_to_all(weights=1.0)

VTARPE = Projection(pre=VTA,post=RPE,target='dopa')
VTARPE.connect_all_to_all(weights=1.0)    

VTAVP_dir = Projection(pre=VTA,post=VP_dir,target='dopa')
VTAVP_dir.connect_all_to_all(weights=1.0)

VTAVP_indir = Projection(pre=VTA,post=VP_indir,target='dopa')
VTAVP_indir.connect_all_to_all(weights=1.0)

# task reward
PPTNVTA = Projection(pre=PPTN,post=VTA,target='exc')
PPTNVTA.connect_one_to_one(weights=1.0)

## handcrafted working memory
ThalObj = Projection(pre=Thal_vent,post=Objectives,target='exc')
ThalObj.connect_one_to_one(weights=1.0) #3.0

ObjStrD1_core = Projection(pre=Objectives,post=StrD1_core,target='exc')
ObjStrD1_core.connect_one_to_one(weights=2.0)    

StrD1SNr_core = Projection(pre=StrD1_core,post=SNr_core,target='inh')
StrD1SNr_core.connect_one_to_one(weights=1.0) #0.75 correction from paper 1.0

SNrThal_core = Projection(pre=SNr_core,post=Thal_vent,target='inh')
SNrThal_core.connect_one_to_one(weights=0.75) #1.2 correction from paper 0.75

ObjStrThal_shell= Projection(pre=Objectives,post=StrThal_shell,target='exc')
ObjStrThal_shell.connect_one_to_one(weights=1.2)


ObjObj = Projection(pre=Objectives,post=Objectives,target='inh')
ObjObj.connect_all_to_all(weights=0.8) 

compile()
