from ANNarchy import (
    Population,
    Neuron,
    Projection,
    Synapse,
    Normal,
    enable_learning,
    simulate,
    compile
)
import optuna
import random

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
Visual_input.tau=10.0
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
dPFC.noise = 0.0

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
DorsomedialBG_SNc.tau = 10.0
DorsomedialBG_SNc.exc_threshold=0.2
DorsomedialBG_SNc.baseline = baseline_dopa_caud
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
DorsomedialBG_Thal.noise = 0.0125
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
ObjStrD1_caud.regularization_threshold = 2.4 #2.4 thesis
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

#CHECK THIS CONNECTION
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

##connect everything besides two first STN 0 and 1 for faces including swap
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
ObjObj = Projection(pre=mPFC,post=mPFC,target='inh')
ObjObj.connect_all_to_all(weights=0.7) #0.75

StrD1StrD1_caud=Projection(pre=DorsomedialBG_StrD1,post=DorsomedialBG_StrD1,target='inh')
StrD1StrD1_caud.connect_all_to_all(weights = 0.2)

StrD2StrD2_caud=Projection(pre=DorsomedialBG_StrD2,post=DorsomedialBG_StrD2,target='inh')
StrD2StrD2_caud.connect_all_to_all(weights= weight_inh_sd2/num_stim)

StrThalStrThal_caud = Projection(pre=DorsomedialBG_cortical_feedback,post=DorsomedialBG_cortical_feedback,target='inh')
StrThalStrThal_caud.connect_all_to_all(weights=weight_inh_thal/num_stim) #0.5

STNSTN_dorsomedial = Projection(pre=DorsomedialBG_STN, post=DorsomedialBG_STN, target='inh')
STNSTN_dorsomedial.connect_all_to_all(weights=0.0)

# inhibition in th  GPi is split to enable learning of both stage choices at the same time
#more neurons turn down excitation
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

ThalThal_caud = Projection(pre=DorsomedialBG_Thal,post=DorsomedialBG_Thal,target='inh')
ThalThal_caud.connect_all_to_all(weights=0.05) #0.15 thesis

ThalThal_caud12 = Projection(pre=DorsomedialBG_Thal[0:2],post=DorsomedialBG_Thal[0:2],target='inh')
ThalThal_caud12.connect_all_to_all(weights=0.15)

ThalThal_caudAB = Projection(pre=DorsomedialBG_Thal[2:],post=DorsomedialBG_Thal[2:],target='inh')
ThalThal_caudAB.connect_all_to_all(weights=0.15)


##DORSOLATERAL LOOP
DorsolatearlBG_StriatumD1= Population(name="StrD1_put", geometry = 6, neuron=LinearNeuron)

DorsolatearlBG_GPi= Population(name="GPi_put", geometry = 2, neuron=LinearNeuron)
DorsolatearlBG_GPi.baseline = 1.25

DorsolatearlBG_Thal= Population(name="Thal_put", geometry = 2, neuron=LinearNeuron)
DorsolatearlBG_Thal.baseline = 0.85

Premotor=Population(name="Premotor", geometry = 2, neuron=LinearNeuron)

# dorsolateral loop
# fixed dorsolateral loop connections
# dorsolateral loop
# fixed dorsolateral loop connections
dPFCStrD1_put=Projection(pre=dPFC,post=DorsolatearlBG_StriatumD1,target="exc")
dPFCStrD1_put.connect_one_to_one(weights=1.0)

StrD1GPi_put1=Projection(pre=DorsolatearlBG_StriatumD1[0:2],post=DorsolatearlBG_GPi,target="inh")
StrD1GPi_put1.connect_one_to_one(weights=1.0)

StrD1GPi_put2=Projection(pre=DorsolatearlBG_StriatumD1[2:4],post=DorsolatearlBG_GPi,target="inh")
StrD1GPi_put2.connect_one_to_one(weights=1.0)

StrD1GPi_put3=Projection(pre=DorsolatearlBG_StriatumD1[4:],post=DorsolatearlBG_GPi,target="inh")
StrD1GPi_put3.connect_one_to_one(weights=1.0)

GPiThal_put=Projection(pre=DorsolatearlBG_GPi,post=DorsolatearlBG_Thal,target="inh")
GPiThal_put.connect_one_to_one(weights=1.0)

ThalPM_put=Projection(pre=DorsolatearlBG_Thal,post=Premotor,target="exc")
ThalPM_put.connect_one_to_one(weights=1.0)

PMPM = Projection(pre=Premotor,post=Premotor,target='inh')
PMPM.connect_all_to_all(weights=0.4) 

compile()



# init the task transition probability 
def objective(trial):
    enable_learning()

    #dorsomedial loop
    tau_ObjStrD1_caud = trial.suggest_int('tau_ObjStrD1_caud',100,600,step=2) #400
    reg_th_ObjStrD1_caud = trial.suggest_float('reg_th_ObjStrD1_caud', 1.0, 5.0, step=0.05) #2.4
    tau_alpha_ObjStrD1_caud = trial.suggest_float('tau_alpha_ObjStrD1_caud', 2.0, 20.0, step=0.1) #5.0
    base_dopa_ObjStrD1_caud = trial.suggest_float('base_dopa_ObjStrD1_caud', 0.1, 2.0, step=0.05) #0.8
    k_dip_ObjStrD1_caud = trial.suggest_float('k_dip_ObjStrD1_caud', 0.0, 2.0, step=0.005) #0.05
    k_burst_ObjStrD1_caud = trial.suggest_float('k_burst_ObjStrD1_caud', 0.0, 4.0, step=0.05) #1.0
    th_pre_ObjStrD1_caud = trial.suggest_float('th_pre_ObjStrD1_caud', 0.0, 3.0, step=0.05) #0.2
    th_post_ObjStrD1_caud = trial.suggest_float('th_post_ObjStrD1_caud', 0.0, 4.0, step=0.05) #0.0

    tau_ObjStrD2_caud = trial.suggest_int('tau_ObjStrD2_caud',500,5000,step=10) #2000.0
    reg_th_ObjStrD2_caud = trial.suggest_float('reg_th_ObjStrD2_caud', 0.0, 5.0, step=0.05) #1.5
    tau_alpha_ObjStrD2_caud = trial.suggest_float('tau_alpha_ObjStrD2_caud', 2.0, 40.0, step=0.5) #15.0
    base_dopa_ObjStrD2_caud = trial.suggest_float('base_dopa_ObjStrD2_caud', 0.1, 2.0, step=0.05) #0.8
    k_dip_ObjStrD2_caud = trial.suggest_float('k_dip_ObjStrD2_caud', 0.0, 2.0, step=0.005) #0.2
    k_burst_ObjStrD2_caud = trial.suggest_float('k_burst_ObjStrD2_caud', 0.0, 4.0, step=0.05) #1.0
    th_pre_ObjStrD2_caud = trial.suggest_float('th_pre_ObjStrD2_caud', 0.0, 0.5, step=0.005) #0.05
    th_post_ObjStrD2_caud = trial.suggest_float('th_post_ObjStrD2_caud', 0.0, 0.5, step=0.005) #0.05

    tau_StrD1GPi = trial.suggest_int('tau_StrD1GPi',500,5000,step=10)  # 1600
    reg_th_StrD1GPi = trial.suggest_float('reg_th_StrD1GPi', 0.0, 5.0, step=0.05) # 2.25
    tau_alpha_StrD1GPi = trial.suggest_float('tau_alpha_StrD1GPi', 2.0, 10.0, step=0.1)  # 4.0 
    base_dopa_StrD1GPi = trial.suggest_float('base_dopa_StrD1GPi', 0.1, 2.0, step=0.05) # 0.8
    k_dip_StrD1GPi = trial.suggest_float('k_dip_StrD1GPi', 0.0, 2.0, step=0.005) # 0.9
    k_burst_StrD1GPi = trial.suggest_float('k_burst_StrD1GPi', 0.0, 4.0, step=0.05) # 1.0
    th_pre_StrD1GPi = trial.suggest_float('th_pre_StrD1GPi', 0.0, 0.5, step=0.005)  # 0.05 
    th_post_StrD1GPi = trial.suggest_float('th_post_StrD1GPi', 0.0, 1.0, step=0.05)  # 0.3 

    tau_StrD1SNc_caud = trial.suggest_int('tau_StrD1SNc_caud',500,5000,step=10)  # 3000

    tau_StrD2GPe = trial.suggest_int('tau_StrD2GPe',500,5000,step=10)  # 2500
    reg_th_StrD2GPe = trial.suggest_float('reg_th_StrD2GPe', 0.0, 5.0, step=0.05) #1.5
    tau_alpha_StrD2GPe = trial.suggest_float('tau_alpha_StrD2GPe', 2.0, 10.0, step=0.1)  #20.0
    base_dopa_StrD2GPe = trial.suggest_float('base_dopa_StrD2GPe', 0.1, 2.0, step=0.05) #0.8
    k_dip_StrD2GPe = trial.suggest_float('k_dip_StrD2GPe', 0.0, 2.0, step=0.005) #0.1
    k_burst_StrD2GPe = trial.suggest_float('k_burst_StrD2GPe', 0.0, 4.0, step=0.05) #1.2
    th_pre_StrD2GPe = trial.suggest_float('th_pre_StrD2GPe', 0.0, 0.5, step=0.005)  # 0.1
    th_post_StrD2GPe = trial.suggest_float('th_post_StrD2GPe', 0.0, 1.0, step=0.05)  #0.0

    w_VISTN_caud = trial.suggest_float('w_VISTN_caud', 0.0, 4.0, step=0.05)

    w_RealSNc_caud = trial.suggest_float('w_RealSNc_caud', 0.0, 4.0, step=0.05) #2.0
    w_RealObj = trial.suggest_float('w_RealObj', 0.0, 4.0, step=0.05) #1.0
    w_GPiThal_caud = trial.suggest_float('w_GPiThal_caud', 0.0, 4.0, step=0.05)
    w_ThalObj_caud = trial.suggest_float('w_ThalObj_caud', 0.0, 4.0, step=0.05)
    w_ObjStrThal_caud = trial.suggest_float('w_ObjStrThal_caud', 0.0, 4.0, step=0.05)
    w_StrThalGPi_caud = trial.suggest_float('w_StrThalGPi_caud', 0.0, 4.0, step=0.05)
    w_ObjObj = trial.suggest_float('w_ObjObj', 0.0, 4.0, step=0.05)
    w_StrD1StrD1_caud = trial.suggest_float('w_StrD1StrD1_caud', 0.0, 4.0, step=0.05)
    w_StrD2StrD2_caud = trial.suggest_float('w_StrD2StrD2_caud', 0.0, 4.0, step=0.05)
    w_StrThalStrThal_caud = trial.suggest_float('w_StrThalStrThal_caud', 0.0, 4.0, step=0.05)
    w_STNSTN_dorsomedial = trial.suggest_float('w_STNSTN_dorsomedial', 0.0, 4.0, step=0.05)
    w_StrThalGPe_caud = trial.suggest_float('w_StrThalGPe_caud', 0.0, 4.0, step=0.05)

    w_GPiGPi = trial.suggest_float('w_GPiGPi', 0.0, 4.0, step=0.05)
    reversal_GPiGPi = trial.suggest_float('reversal_GPiGPi', 0.0, 4.0, step=0.05)

    w_GPiGPi12 = trial.suggest_float('w_GPiGPi12', 0.0, 4.0, step=0.05)
    reversal_GPiGPi12 = trial.suggest_float('reversal_GPiGPi12', 0.0, 4.0, step=0.05)

    w_GPiGPiAB = trial.suggest_float('w_GPiGPiAB', 0.0, 4.0, step=0.05)
    reversal_GPiGPiAB = trial.suggest_float('reversal_GPiGPiAB', 0.0, 4.0, step=0.05)

    w_dPFCdPFC = trial.suggest_float('w_dPFCdPFC', 0.0, 4.0, step=0.05)
    w_dPFCdPFC12 = trial.suggest_float('w_dPFCdPFC12', 0.0, 4.0, step=0.05)
    w_dPFCdPFCAB = trial.suggest_float('w_dPFCdPFCAB', 0.0, 4.0, step=0.05)

    w_ThalThal_caud = trial.suggest_float('w_ThalThal_caud', 0.0, 2.0, step=0.005)
    w_ThalThal_caud12 = trial.suggest_float('w_ThalThal_caud12', 0.0, 4.0, step=0.05)
    w_ThalThal_caudAB = trial.suggest_float('w_ThalThal_caudAB', 0.0, 4.0, step=0.05)

    #dorsomedial loop
    ObjStrD1_caud.tau =  tau_ObjStrD1_caud
    ObjStrD1_caud.regularization_threshold = reg_th_ObjStrD1_caud
    ObjStrD1_caud.tau_alpha = tau_alpha_ObjStrD1_caud
    ObjStrD1_caud.baseline_dopa =  base_dopa_ObjStrD1_caud
    ObjStrD1_caud.K_dip =  k_dip_ObjStrD1_caud
    ObjStrD1_caud.K_burst = k_burst_ObjStrD1_caud
    ObjStrD1_caud.threshold_pre = th_pre_ObjStrD1_caud
    ObjStrD1_caud.threshold_post = th_post_ObjStrD1_caud

    ObjStrD2_caud.tau = tau_ObjStrD2_caud
    ObjStrD2_caud.regularization_threshold =reg_th_ObjStrD2_caud
    ObjStrD2_caud.tau_alpha = tau_alpha_ObjStrD2_caud
    ObjStrD2_caud.baseline_dopa =base_dopa_ObjStrD2_caud
    ObjStrD2_caud.K_dip =k_dip_ObjStrD2_caud
    ObjStrD2_caud.K_burst =k_burst_ObjStrD2_caud
    ObjStrD2_caud.threshold_pre = th_pre_ObjStrD2_caud
    ObjStrD2_caud.threshold_post = th_post_ObjStrD2_caud

    #CHECK THIS CONNECTION
    StrD1GPi.tau =  tau_StrD1GPi
    StrD1GPi.regularization_threshold =  reg_th_StrD1GPi
    StrD1GPi.tau_alpha =  tau_alpha_StrD1GPi
    StrD1GPi.baseline_dopa = base_dopa_StrD1GPi
    StrD1GPi.K_dip = k_dip_StrD1GPi
    StrD1GPi.K_burst = k_burst_StrD1GPi
    StrD1GPi.threshold_pre =  th_pre_StrD1GPi
    StrD1GPi.threshold_post = th_post_StrD1GPi

    StrD1SNc_caud.tau = tau_StrD1SNc_caud

    StrD2GPe.tau =  tau_StrD2GPe
    StrD2GPe.regularization_threshold = reg_th_StrD2GPe
    StrD2GPe.tau_alpha =  tau_alpha_StrD2GPe
    StrD2GPe.baseline_dopa = base_dopa_StrD2GPe
    StrD2GPe.K_dip = k_dip_StrD2GPe
    StrD2GPe.K_burst = k_burst_StrD2GPe
    StrD2GPe.threshold_pre = th_pre_StrD2GPe
    StrD2GPe.threshold_post = th_post_StrD2GPe

    #Input is incorporated via the fixed hyperdirect pathway
    for pop in VISTN_caud:
        pop.w = w_VISTN_caud

    RealSNc_caud.w = w_RealSNc_caud
    RealObj.w = w_RealObj
    GPiThal_caud.w = w_GPiThal_caud
    ThalObj_caud.w = w_ThalObj_caud
    ObjStrThal_caud.w = w_ObjStrThal_caud
    StrThalGPi_caud.w = w_StrThalGPi_caud
    StrThalGPe_caud.w = w_StrThalGPe_caud
    ObjObj.w = w_ObjObj
    StrD1StrD1_caud.w = w_StrD1StrD1_caud
    StrD2StrD2_caud.w = w_StrD2StrD2_caud
    StrThalStrThal_caud.w = w_StrThalStrThal_caud
    STNSTN_dorsomedial.w = w_STNSTN_dorsomedial

    GPiGPi.w = w_GPiGPi
    GPiGPi.reversal = reversal_GPiGPi

    GPiGPi12.w = w_GPiGPi12
    GPiGPi12.reversal = reversal_GPiGPi12

    GPiGPiAB.w = w_GPiGPiAB
    GPiGPiAB.reversal = reversal_GPiGPiAB


    dPFCdPFC.w = w_dPFCdPFC
    dPFCdPFC12.w = w_dPFCdPFC12
    dPFCdPFCAB.w = w_dPFCdPFCAB

    ThalThal_caud.w = w_ThalThal_caud
    ThalThal_caud12.w = w_ThalThal_caud12
    ThalThal_caudAB.w = w_ThalThal_caudAB

    random.seed()
    num_trials = 250
    eps=0.000001

    # initialize some arrays for the results
    realizable_stimulus=[]

    selected_obj_counter = 0
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
        simulate(200)

        selected_objective=random.choices(list(range(8)),weights=(mPFC.r+eps),k=1)[0]
        
        mPFC[selected_objective].baseline = 0.8

        simulate(300)

        # make the first decision according to Premotor neurons
        motor_objectives=Premotor.r
        #first action selected: left or right -> (0,1)
        first_action=random.choices([0,1],weights=motor_objectives+eps,k=1)[0]

        Visual_input.baseline=0.0

        #action 1
        if first_action==0:
            #common transition
            if random.random()<transition_prob:
                Visual_input[:,1].baseline=1.5
                second_panel=1
            #rare transition
            else:
                Visual_input[:,2].baseline=1.5
                second_panel=2
        #action 2
        if first_action==1:
            #common transition
            if random.random()<transition_prob:
                Visual_input[:,2].baseline=1.5
                second_panel=2
            #rare transition
            else:
                Visual_input[:,1].baseline=1.5
                second_panel=1

        # second_panel = 1 -> (A,B)
        # second_panel = 2 -> (C,D)
        #{0:"(1,A)",1:"(1,B)",2:"(1,C)",3:"(1,D)",4:"(2,A)",5:"(2,B)",6:"(2,C)",7:"(2,D)"}
        #check if stimulis proposed by the mPFC is doable
        #not doable when the objective isn't in the second panel
        #if second panel is 2 (C,D) sequences 0,1,4,5 are not possible
        realizable = True
        if (second_panel==1 and (selected_objective in [0,1,4,5])) or (second_panel==2 and selected_objective in [2,3,6,7]):
            realizable_stimulus.append(True)
        else:
            realizable_stimulus.append(False)
        
        simulate(300)

        # decision of the Wm is reinforced, as no other sequences are shown by the hippocampus
        DorsomedialBG_cortical_feedback.learning=1.0   
        simulate(200)
        #second decision, choose between options from the second panel (left or right, same as before)
        motor_objectives=Premotor.r
        second_action=random.choices([0,1],weights=motor_objectives+eps,k=1)[0]

        #second action is in the range [0,1,2,3] -> [A,B,C,D]
        if second_panel==2:
            second_action+=2

        #calculate the sequence selected so it's in the range [0,7]
        #eg. if sequence is 5 -> (2,B) this means that first and second action are 1 and 1 respectively
        #that gives a total of 4*1*+1 = 5, you can calculate the others to check
        #print("first_action", first_action, "second_action", second_action)
        achieved_sequence=4*first_action+second_action

        ## LEARNING PART
        # integration of the realizied objective into the cortex -> a bit weird as we we have to circumvent very high PFC activities (otherwise unlearning because of regularization with alpha)     
        DorsomedialBG_PPTN[achieved_sequence].baseline=2.0
        Visual_input.baseline=0
        #update columns for the selected sequence
        #update dPFC
        mPFC[achieved_sequence].baseline=2.0
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

        #allow weight changes
        DorsomedialBG_SNc.firing=1

        #actual learning period, outcome
        simulate(100)

        DorsomedialBG_SNc.firing=0

        if selected_objective==achieved_sequence:
            selected_obj_counter += 1

    correct_proportion = selected_obj_counter/sum(realizable_stimulus)
    objective = 1.0

    loss = objective - correct_proportion
    print(
        "total objective reached", selected_obj_counter,
          "out of", sum(realizable_stimulus),
           "meaning", (selected_obj_counter/sum(realizable_stimulus))*100,"%")
    
    return loss

study = optuna.create_study(
    study_name='daw_dorsomedial',
    storage='mysql://optuna:123456@localhost:3306/optuna',
    load_if_exists=True
)
study.optimize(objective, n_trials=10)
print(study.best_params)

