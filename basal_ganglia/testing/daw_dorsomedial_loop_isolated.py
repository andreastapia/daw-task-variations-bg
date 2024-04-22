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
import numpy as np
import matplotlib.pyplot as plt

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
weight_local_inh = 0.8 *0.2 * 0.166 #old model (Scholl) with 6 distinct population -> 0.8*0.2, here only one -> scale it by 1/6
weight_stn_inh = 0.3 
weight_inh_sd2 = 0.5 *0.2
weight_inh_thal = 0.5 *0.2
modul = 0.01

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

DAPostCovarianceNoThreshold_trace = Synapse(
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
        trace = pos(post.trace -  mean(post.trace) - threshold_post) * (pre.r - mean(pre.r) - threshold_pre)
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

DAPreCovariance_inhibitory_trace = Synapse(
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
    equations="""
        tau_alpha*dalpha/dt = pos( -post.mp - regularization_threshold) - alpha
        dopa_sum = 2.0*(post.sum(dopa) - baseline_dopa)
        trace = pos(pre.trace - mean(pre.trace) - threshold_pre) * (mean(post.r) - post.r  - threshold_post)
        aux = if (trace>0): 1 else: 0
        dopa_mod = if (DA_type*dopa_sum>0): K_burst*dopa_sum else: aux*DA_type*K_dip*dopa_sum
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

ReversedSynapse = Synapse(
    parameters="""
        reversal = 0.3
    """,
    psp="""
        w*pos(reversal-pre.r)
    """
)

# ~~~~~~~~~~~~ DORSOMEDIAL LOOP (ACTION SELECTION) ~~~~~~~~~~~~
# dorsomedial loop (caudate)
#shared with ventral loop
Visual_input = Population(name="Visual_input",geometry=(4,3), neuron=LinearNeuron)
Visual_input.baseline=0.0
Visual_input.noise=0.001

# Neurons for the ventral loop, previously called "Objectives"
#mPFC, is shared with working memory and ventral loop
mPFC = Population(name="mPFC_Objectives", geometry=num_objectives, neuron=LinearNeuron)
mPFC.tau = 40.0 # 40
mPFC.noise = 0.01

#dorsal prefrontal cortex
#shared with dorsolateral loop
dPFC = Population(name="dPFC", geometry=6, neuron=LinearNeuron)
dPFC.tau = 40.0 # 40
dPFC.noise = 0.01 #curr: 0.01 thesis: 0.0

#previously called STN_caud
DorsomedialBG_STN = Population(name="DorsomedialBG_STN", geometry = (3), neuron=LinearNeuron)
DorsomedialBG_STN.tau = 10.0
DorsomedialBG_STN.noise = 0.01
DorsomedialBG_STN.baseline = 0.0

#previously called StrD1_caud
DorsomedialBG_StrD1 = Population(name="DorsomedialBG_StrD1", geometry=(6, 8),neuron = LinearNeuron_trace)
DorsomedialBG_StrD1.tau = 10.0
DorsomedialBG_StrD1.noise = 0.4
DorsomedialBG_StrD1.baseline = 0.0
DorsomedialBG_StrD1.lesion = 1.0
DorsomedialBG_StrD1.tau_trace = gen_tau_trace

#previously called StrD2_caud
DorsomedialBG_StrD2 = Population(name="DorsomedialBG_StrD2", geometry = (3, 8), neuron=LinearNeuron_trace)
DorsomedialBG_StrD2.tau = 10.0
DorsomedialBG_StrD2.noise = 0.01
DorsomedialBG_StrD2.baseline = 0.0
DorsomedialBG_StrD2.lesion = 1.0

#previously called SNc_caud
DorsomedialBG_SNc = Population(name='DorsomedialBG_SNc',geometry=8,neuron=DopamineNeuron)
DorsomedialBG_SNc.exc_threshold=0.2
DorsomedialBG_SNc.baseline = baseline_dopa_caud #0.1
DorsomedialBG_SNc.factor_inh = 1.0

# acts as the Propio population in Scholl 2021
# PPTN in dorsomedial loop, previously called Realized_sequence
DorsomedialBG_PPTN = Population(name="DorsomedialBG_PPTN",geometry=8,neuron=InputNeuron)

GPe = Population(name="GPe", geometry = 6, neuron=LinearNeuron)
GPe.tau = 10.0
GPe.noise = 0.05
GPe.baseline = 1.0

GPi = Population(name="GPi", geometry = 6, neuron=LinearNeuron)
GPi.tau = 10.0
GPi.noise = 0.5 #0.3
GPi.baseline = 1.9 #0.75 #1.5


#previously called Thal_caud
DorsomedialBG_Thal = Population(name="DorsomedialBG_Thal", geometry=6, neuron=LinearNeuron)
DorsomedialBG_Thal.tau = 10.0
DorsomedialBG_Thal.noise = 0.025
DorsomedialBG_Thal.baseline = 0.7

#cortical feedback for dorsomedial loop
#previously called StrThal_caud
DorsomedialBG_cortical_feedback = Population(name="DorsomedialBG_cortical_feedback", geometry = 6, neuron=LinearNeuron_learning)
DorsomedialBG_cortical_feedback.tau = 10.0
DorsomedialBG_cortical_feedback.noise = 0.01
DorsomedialBG_cortical_feedback.baseline = 0.4
DorsomedialBG_cortical_feedback.lesion = 1.0
DorsomedialBG_cortical_feedback.learning=1.0

#dorsomedial loop
ObjStrD1_caud = Projection(pre=mPFC,post=DorsomedialBG_StrD1,target='exc',synapse=DAPostCovarianceNoThreshold_trace) 
ObjStrD1_caud.connect_all_to_all(weights = Normal(0.5,0.2))
ObjStrD1_caud.tau = 400 #100
ObjStrD1_caud.regularization_threshold = 2.0 #curr: 2.0 thesis: 2.4
ObjStrD1_caud.tau_alpha = 5.0
ObjStrD1_caud.baseline_dopa = 8*baseline_dopa_caud
ObjStrD1_caud.K_dip = 0.05
ObjStrD1_caud.K_burst = 1.0
ObjStrD1_caud.DA_type = 1
ObjStrD1_caud.threshold_pre = 0.2
ObjStrD1_caud.threshold_post = 0.0

ObjStrD2_caud = Projection(pre=mPFC,post=DorsomedialBG_StrD2,target='exc',synapse=DAPostCovarianceNoThreshold)
ObjStrD2_caud.connect_all_to_all(weights = Normal(0.12,0.03)) #0.005
ObjStrD2_caud.tau = 2000.0
ObjStrD2_caud.regularization_threshold = 1.5
ObjStrD2_caud.tau_alpha = 15.0
ObjStrD2_caud.baseline_dopa = 8*baseline_dopa_caud
ObjStrD2_caud.K_dip = 0.2
ObjStrD2_caud.K_burst = 1.0
ObjStrD2_caud.DA_type = -1
ObjStrD2_caud.threshold_pre = 0.05
ObjStrD2_caud.threshold_post = 0.05

StrD1GPi = Projection(pre=DorsomedialBG_StrD1,post=GPi,target='inh',synapse=DAPreCovariance_inhibitory_trace)
StrD1GPi.connect_all_to_all(weights=Normal(0.1,0.01)) # scale by numer of stimuli #var 0.01
StrD1GPi.tau = 1600 #700 #550
StrD1GPi.regularization_threshold = 2.25 #1.5
StrD1GPi.tau_alpha = 4.0 # 20.0
StrD1GPi.baseline_dopa = 8*baseline_dopa_caud
StrD1GPi.K_dip = 0.9
StrD1GPi.K_burst = 1.0
StrD1GPi.threshold_post = 0.3 #0.1 #0.3
StrD1GPi.threshold_pre = 0.05 # 0.15
StrD1GPi.DA_type=1
StrD1GPi.negterm = 5.0

# dopamine prediction by the StriatumD1 neurons (as typical)
StrD1SNc_caud = Projection(pre=DorsomedialBG_StrD1,post=DorsomedialBG_SNc,target='inh',synapse=DAPrediction)
StrD1SNc_caud.connect_all_to_all(weights=0.0)
StrD1SNc_caud.tau = 3000

StrD2GPe = Projection(pre=DorsomedialBG_StrD2,post=GPe,target='inh',synapse=DAPreCovariance_inhibitory)
StrD2GPe.connect_all_to_all(weights=(0.01)) # was 0.01
StrD2GPe.tau = 2500
StrD2GPe.regularization_threshold = 1.5
StrD2GPe.tau_alpha = 20.0
StrD2GPe.baseline_dopa = 8*baseline_dopa_caud
StrD2GPe.K_dip = 0.1
StrD2GPe.K_burst = 1.2
StrD2GPe.threshold_post = 0.0
StrD2GPe.threshold_pre = 0.1
StrD2GPe.DA_type = -1

#Input is incorporated via the fixed hyperdirect pathway
VISTN_caud=[]

for i in range(3):
    VISTN_caud.append(Projection(pre=Visual_input[:,i],post=DorsomedialBG_STN[i],target="exc"))
    VISTN_caud[i].connect_all_to_all(weights=1.0/4.0)

Panel1 = Projection(pre=DorsomedialBG_STN[0],post=GPi[2:],target="exc")
Panel1.connect_all_to_all(weights=1.7)
Panel2=[]
Panel2.append(Projection(pre=DorsomedialBG_STN[1],post=GPi[:2],target="exc"))
Panel2[-1].connect_all_to_all(weights=1.7)
Panel2.append(Projection(pre=DorsomedialBG_STN[1],post=GPi[4:],target="exc"))
Panel2[-1].connect_all_to_all(weights=1.7)
Panel3 = Projection(pre=DorsomedialBG_STN[2],post=GPi[:4],target="exc")
Panel3.connect_all_to_all(weights=1.7)

RealSNc_caud = Projection(pre=DorsomedialBG_PPTN,post=DorsomedialBG_SNc,target="exc")
RealSNc_caud.connect_one_to_one(weights=2.0)

RealObj= Projection(pre=DorsomedialBG_PPTN,post=mPFC,target="exc")
RealObj.connect_one_to_one(weights=1.0)

# dopamine encoded in seperate population (compared to ventral loop)
SNcStrD1_caud = Projection(pre=DorsomedialBG_SNc,post=DorsomedialBG_StrD1,target='dopa')
SNcStrD1_caud.connect_all_to_all(weights=1.0)

SNcStrD2_caud = Projection(pre=DorsomedialBG_SNc,post=DorsomedialBG_StrD2,target='dopa')
SNcStrD2_caud.connect_all_to_all(weights=1.0)

SNcSTN_caud = Projection(pre=DorsomedialBG_SNc,post=DorsomedialBG_STN,target='dopa')
SNcSTN_caud.connect_all_to_all(weights=1.0)

SNcGPi_caud = Projection(pre=DorsomedialBG_SNc,post=GPi,target='dopa')
SNcGPi_caud.connect_all_to_all(weights=1.0)

SNcGPe_caud = Projection(pre=DorsomedialBG_SNc,post=GPe,target='dopa')
SNcGPe_caud.connect_all_to_all(weights=1.0)

GPeGPi = Projection(pre=GPe,post=GPi,target='inh')
GPeGPi.connect_one_to_one(weights=1.0) 

GPiThal_caud = Projection(pre=GPi,post=DorsomedialBG_Thal,target='inh')
GPiThal_caud.connect_one_to_one(weights=1.7)

ThalObj_caud = Projection(pre=DorsomedialBG_Thal,post=dPFC,target='exc')
ThalObj_caud.connect_one_to_one(weights=1.5)

ObjStrThal_caud= Projection(pre=dPFC,post=DorsomedialBG_cortical_feedback,target='exc')
ObjStrThal_caud.connect_one_to_one(weights=1.2)

StrThalGPi_caud = Projection(pre=DorsomedialBG_cortical_feedback,post=GPi,target='inh')
StrThalGPi_caud.connect_one_to_one(weights=0.8)

StrThalGPe_caud = Projection(pre=DorsomedialBG_cortical_feedback,post=GPe,target='inh')
StrThalGPe_caud.connect_one_to_one(weights=0.3)

#inhibitory connections
STNSTN_dorsomedial = Projection(pre=DorsomedialBG_STN, post=DorsomedialBG_STN, target='inh')
STNSTN_dorsomedial.connect_all_to_all(weights=0.0)

ObjObj = Projection(pre=mPFC,post=mPFC,target='inh')
ObjObj.connect_all_to_all(weights=0.8) 

StrD1StrD1_caud=Projection(pre=DorsomedialBG_StrD1,post=DorsomedialBG_StrD1,target='inh')
StrD1StrD1_caud.connect_all_to_all(weights = 0.2)

StrD2StrD2_caud=Projection(pre=DorsomedialBG_StrD2,post=DorsomedialBG_StrD2,target='inh')
StrD2StrD2_caud.connect_all_to_all(weights=weight_inh_sd2/num_stim) #0.016666

StrThalStrThal_caud = Projection(pre=DorsomedialBG_cortical_feedback,post=DorsomedialBG_cortical_feedback,target='inh')
StrThalStrThal_caud.connect_all_to_all(weights=weight_inh_thal/num_stim) 

# inhibition in th  GPi is split to enable learning of both stage choices at the same time
GPiGPi = Projection(pre=GPi,post=GPi,target='exc',synapse=ReversedSynapse)
GPiGPi.connect_all_to_all(weights=0.15)
GPiGPi.reversal = 0.4

GPiGPi12 = Projection(pre=GPi[0:2],post=GPi[0:2],target='exc',synapse=ReversedSynapse)
GPiGPi12.connect_all_to_all(weights=0.35)
GPiGPi12.reversal = 0.4

GPiGPiAB = Projection(pre=GPi[2:],post=GPi[2:],target='exc',synapse=ReversedSynapse)
GPiGPiAB.connect_all_to_all(weights=0.35)
GPiGPiAB.reversal = 0.4

dPFCdPFC = Projection(pre=dPFC,post=dPFC,target='inh')
dPFCdPFC.connect_all_to_all(weights=0.2)

dPFCdPFC12 = Projection(pre=dPFC[0:2],post=dPFC[0:2],target='inh')
dPFCdPFC12.connect_all_to_all(weights=0.5)

dPFCdPFCAB = Projection(pre=dPFC[2:],post=dPFC[2:],target='inh')
dPFCdPFCAB.connect_all_to_all(weights=0.5)

ThalThal_caud = Projection(pre=DorsomedialBG_Thal[0:2],post=DorsomedialBG_Thal[0:2],target='inh')
ThalThal_caud.connect_all_to_all(weights=0.05) #curr: 0.05 thesis: 0.15

ThalThal_caud12 = Projection(pre=DorsomedialBG_Thal[0:2],post=DorsomedialBG_Thal[0:2],target='inh')
ThalThal_caud12.connect_all_to_all(weights=0.15)

ThalThal_caudAB = Projection(pre=DorsomedialBG_Thal[2:],post=DorsomedialBG_Thal[2:],target='inh')
ThalThal_caudAB.connect_all_to_all(weights=0.15)

compile()

#enable learning for all projections
def diffuse_prob(prob):
    """Diffuses a probability between 0.25 and 0.75"""
    prob += random.gauss(0, 0.025)
    if prob < 0.25:
        prob = 0.5 - prob
    elif prob > 0.75:
        prob = 1.5 - prob
    assert prob >= 0.25 and prob <= 0.75
    return prob

enable_learning()
random.seed()
num_trials = 150
eps=0.000001
stimuli_history=[]
possible_first_actions = [0,1]

#keys for the actions
stimulis=list(range(8))
stim_to_action={0:"(1,A)",1:"(1,B)",2:"(1,C)",3:"(1,D)",4:"(2,A)",5:"(2,B)",6:"(2,C)",7:"(2,D)"}

realizable_stimulus=[]
#stores info about the trials
common_transition=[]
rewarded=[]
rewards=[]
second_actions=[]


# set monitor variables for output of important activities, this records variables or populations
monitor_thal=Monitor(DorsomedialBG_Thal,'r')
monitor_thal.pause()
monitor_mpfc=Monitor(mPFC,'r')
monitor_mpfc.pause()
monitor_gpi=Monitor(GPi,'r')
monitor_gpi.pause()
monitor_dpfc=Monitor(dPFC,'r')
monitor_dpfc.pause()
monitor_gpi_learning=Monitor(GPi,'r')
monitor_gpi_learning.pause()
monitor_dpfc_learning=Monitor(dPFC,'r')
monitor_dpfc_learning.pause()
monitor_objective_learning=Monitor(mPFC,'r')
monitor_objective_learning.pause()

gpi_activities=[]
dPFC_activities=[]
mpfc_activities=[]
thal_activities=[]
objective_learning=[]
gpi_learning=[]
dpfc_learning=[]

# initialize some arrays for the results
selected_objectives=[]
achieved_sequences=[]
reward_of_actions=[0.75, 0.25, 0.25, 0.25]
dopamine_activities=[]
achieved_desired_sequence=[]
acc_reward=[0]

# init the task transition probability 
transition_prob=0.7

for trial in range(num_trials):
    mPFC.r = 0.0
    mPFC.baseline = 0.0

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

    DorsomedialBG_Thal.baseline=0.95
    Visual_input[:,0].baseline=1.5
    # reset finished

    monitor_mpfc.resume()
    monitor_thal.resume()
    monitor_gpi.resume()
    monitor_dpfc.resume()

    selected_objective=random.choices(list(range(8)),weights=(mPFC.r+eps),k=1)[0]
    selected_objectives.append(selected_objective)
    
    mPFC[selected_objective].baseline = 2.0

    simulate(300)

    first_action = selected_objective//4
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
    
    simulate(300)
    
    realizable = False
    if (second_panel==1 and (selected_objective in [0,1,4,5])) or (second_panel==2 and selected_objective in [2,3,6,7]):
        realizable_stimulus.append(True)
        realizable = True
    else:
        realizable_stimulus.append(False)

    #some time for second decision -> can be shorter if one wants that
    simulate(200)

    # decision of the Wm is reinforced, as no other sequences are shown by the hippocampus
    DorsomedialBG_cortical_feedback.learning=1.0   
    simulate(200)
    second_action = selected_objective%4
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
    # integration of the realizied objective into the cortex -> a bit weird as we we have to circumvent very high PFC activities (otherwise unlearning because of regularization with alpha)
    DorsomedialBG_PPTN[achieved_sequence].baseline=2.0
    Visual_input.baseline=0
    #update columns for the selected sequence
    mPFC[achieved_sequence].baseline=2.0
    #update dPFC
    dPFC[first_action].baseline=2.0
    dPFC[2+second_action].baseline=2.0

    # first short step with higer intergration
    simulate(100)

    # now reduce the feedback strenght
    dPFC[first_action].baseline=1.5
    dPFC[2+second_action].baseline=1.5
    mPFC[achieved_sequence].baseline=0.2
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
        acc_reward_new_entry+=1
        rewarded.append(1)
        rew = 1
    #non-rewarded
    else:
        rewarded.append(0)
        acc_reward.append(acc_reward_new_entry)

    #allow weight changes
    DorsomedialBG_SNc.firing=1

    #actual learning period, outcome
    simulate(100)

    DorsomedialBG_SNc.firing=0

    second_actions.append(second_action)
    monitor_mpfc.pause()
    monitor_thal.pause()
    monitor_gpi.pause()
    monitor_dpfc.pause() 

    dPFC_activities.append(monitor_dpfc.get("r"))
    gpi_activities.append(monitor_gpi.get("r"))
    thal_activities.append(monitor_thal.get("r"))
    mpfc_activities.append(monitor_mpfc.get("r"))

    #save activities of important nuclei during certain trials 
    # if trial in [0,10,20,40,60,80]:
    #     trial_rates_path = result_path + "rates/trial_" + str(trial)
    #     saveAllActivations(trial_rates_path)

    #this stores if the sequence proposed by the mPFC is actually selected by the other loops
    #this is not sored, you could
    achieved_desired_sequence.append(selected_objective==achieved_sequence)

    rewards.insert(trial, reward_of_actions.copy())

    print(
    "obj", selected_objective,
    "ach", achieved_sequence,
    "first",first_action, 
    "second",second_action,
    "rewarded", False, #rew,
    "trial", trial,
    "realizable", realizable)

    # change the rewards
    for i in range(4):
        reward_of_actions[i]=diffuse_prob(reward_of_actions[i])

rewards=np.array(rewards)    
achieved_desired_sequence=np.array(achieved_desired_sequence)

mpfc_act_np = np.array(mpfc_activities)
gpi_act_np = np.array(gpi_activities)
dfpc_act_np = np.array(dPFC_activities)
objective_learning_act_np = np.array(objective_learning)
gpi_learning_act_np = np.array(gpi_learning)
dpfc_learning_act_np = np.array(dpfc_learning)

trial_pick = 100

fig, (ax1, ax2, ax3) = plt.subplots(3)

ax1.plot(np.arange(mpfc_act_np.shape[1]), mpfc_act_np[trial_pick][:,0], label='(1,A)')
ax1.plot(np.arange(mpfc_act_np.shape[1]), mpfc_act_np[trial_pick][:,1], label='(1,B)')
ax1.plot(np.arange(mpfc_act_np.shape[1]), mpfc_act_np[trial_pick][:,2], label='(1,C)')
ax1.plot(np.arange(mpfc_act_np.shape[1]), mpfc_act_np[trial_pick][:,3], label='(1,D)')
ax1.plot(np.arange(mpfc_act_np.shape[1]), mpfc_act_np[trial_pick][:,4], label='(2,A)')
ax1.plot(np.arange(mpfc_act_np.shape[1]), mpfc_act_np[trial_pick][:,5], label='(2,B)')
ax1.plot(np.arange(mpfc_act_np.shape[1]), mpfc_act_np[trial_pick][:,6], label='(2,C)')
ax1.plot(np.arange(mpfc_act_np.shape[1]), mpfc_act_np[trial_pick][:,7], label='(2,D)')

ax2.plot(np.arange(gpi_act_np.shape[1]), gpi_act_np[trial_pick][:,0], label='(1)')
ax2.plot(np.arange(gpi_act_np.shape[1]), gpi_act_np[trial_pick][:,1], label='(2)')
ax2.plot(np.arange(gpi_act_np.shape[1]), gpi_act_np[trial_pick][:,2], label='(A)')
ax2.plot(np.arange(gpi_act_np.shape[1]), gpi_act_np[trial_pick][:,3], label='(B)')
ax2.plot(np.arange(gpi_act_np.shape[1]), gpi_act_np[trial_pick][:,4], label='(C)')
ax2.plot(np.arange(gpi_act_np.shape[1]), gpi_act_np[trial_pick][:,5], label='(D)')

ax3.plot(np.arange(dfpc_act_np.shape[1]), dfpc_act_np[trial_pick][:,0], label='(1)')
ax3.plot(np.arange(dfpc_act_np.shape[1]), dfpc_act_np[trial_pick][:,1], label='(2)')
ax3.plot(np.arange(dfpc_act_np.shape[1]), dfpc_act_np[trial_pick][:,2], label='(A)')
ax3.plot(np.arange(dfpc_act_np.shape[1]), dfpc_act_np[trial_pick][:,3], label='(B)')
ax3.plot(np.arange(dfpc_act_np.shape[1]), dfpc_act_np[trial_pick][:,4], label='(C)')
ax3.plot(np.arange(dfpc_act_np.shape[1]), dfpc_act_np[trial_pick][:,5], label='(D)')

# ax1.plot(np.arange(gpi_learning_act_np.shape[1]), gpi_learning_act_np[trial_pick][:,0], label='(1)')
# ax1.plot(np.arange(gpi_learning_act_np.shape[1]), gpi_learning_act_np[trial_pick][:,1], label='(2)')
# ax1.plot(np.arange(gpi_learning_act_np.shape[1]), gpi_learning_act_np[trial_pick][:,2], label='(A)')
# ax1.plot(np.arange(gpi_learning_act_np.shape[1]), gpi_learning_act_np[trial_pick][:,3], label='(B)')
# ax1.plot(np.arange(gpi_learning_act_np.shape[1]), gpi_learning_act_np[trial_pick][:,4], label='(C)')
# ax1.plot(np.arange(gpi_learning_act_np.shape[1]), gpi_learning_act_np[trial_pick][:,5], label='(D)')

# ax2.plot(np.arange(dpfc_learning_act_np.shape[1]), dpfc_learning_act_np[trial_pick][:,0], label='(1)')
# ax2.plot(np.arange(dpfc_learning_act_np.shape[1]), dpfc_learning_act_np[trial_pick][:,1], label='(2)')
# ax2.plot(np.arange(dpfc_learning_act_np.shape[1]), dpfc_learning_act_np[trial_pick][:,2], label='(A)')
# ax2.plot(np.arange(dpfc_learning_act_np.shape[1]), dpfc_learning_act_np[trial_pick][:,3], label='(B)')
# ax2.plot(np.arange(dpfc_learning_act_np.shape[1]), dpfc_learning_act_np[trial_pick][:,4], label='(C)')
# ax2.plot(np.arange(dpfc_learning_act_np.shape[1]), dpfc_learning_act_np[trial_pick][:,5], label='(D)')

# ax3.plot(np.arange(objective_learning_act_np.shape[1]), objective_learning_act_np[trial_pick][:,0], label='(1)')
# ax3.plot(np.arange(objective_learning_act_np.shape[1]), objective_learning_act_np[trial_pick][:,1], label='(2)')
# ax3.plot(np.arange(objective_learning_act_np.shape[1]), objective_learning_act_np[trial_pick][:,2], label='(A)')
# ax3.plot(np.arange(objective_learning_act_np.shape[1]), objective_learning_act_np[trial_pick][:,3], label='(B)')
# ax3.plot(np.arange(objective_learning_act_np.shape[1]), objective_learning_act_np[trial_pick][:,4], label='(C)')
# ax3.plot(np.arange(objective_learning_act_np.shape[1]), objective_learning_act_np[trial_pick][:,5], label='(D)')

ax1.legend(loc='upper right', title='Secuencia')
ax3.legend(loc='upper right', title='Opcion')
fig.tight_layout()
plt.show()