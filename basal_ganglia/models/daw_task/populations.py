from ANNarchy import Population
from .neurons import (
    LinearNeuron,
    InputNeuron,
    LinearNeuron_trace,
    LinearNeuron_learning,
    DopamineNeuron
)
#all parameters are used except for the neurons_per_goal
from .params import *

# ~~~~~~~~~~~~ VENTRAL LOOP (SEQUENCE SELECTION) ~~~~~~~~~~~~

# Neurons for the ventral loop, previously called "Objectives"
#mPFC, is shared with working memory and ventral loop
mPFC = Population(name="mPFC_Objectives", geometry=num_objectives, neuron=LinearNeuron)
mPFC.tau = 40.0 # 40
mPFC.noise = 0.01

# Hippocampus is implemented as Input neurons cycling through the different sequences 
# Used on the ventral loop
# previously called Input_Neurons
Hippocampus = Population(name='Hipp_Input_Neurons',geometry=(4,num_stim),neuron=InputNeuron)

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

# STN
VentralBG_STN = Population(name="VentralBG_STN_shell", geometry = (4, num_stim), neuron=LinearNeuron_trace)
VentralBG_STN.tau = 10.0
VentralBG_STN.noise = 0.01
VentralBG_STN.baseline = 0.0

# enables hebbian learning rule by excitation of the VP neurons
# previously called StrThal_shell
VentralBG_Cortical_feedback = Population(name="VentralBG_Cortical_feedback", geometry = num_objectives, neuron=LinearNeuron_learning)
VentralBG_Cortical_feedback.tau = 10.0
VentralBG_Cortical_feedback.noise = 0.01
VentralBG_Cortical_feedback.baseline = 0.4
VentralBG_Cortical_feedback.lesion = 1.0

# Thal for WM/ventral loop, previously called Thal_vent
VentralBG_Thal = Population(name="VentralBG_Thal", geometry=num_objectives, neuron=LinearNeuron)
VentralBG_Thal.tau = 10.0
VentralBG_Thal.noise = 0.025 / 2.0
VentralBG_Thal.baseline = 1.2

# Reward cells of shell
VTA = Population(name='VTA',geometry=1,neuron=DopamineNeuron)
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

#~~~~~~~~~~~~~~~DORSOLATERAL LOOP~~~~~~~~~~~~~~~~~~
DorsolatearlBG_StriatumD1= Population(name="StrD1_put", geometry = 6, neuron=LinearNeuron)
DorsolatearlBG_GPi= Population(name="GPi_put", geometry = 2, neuron=LinearNeuron)
DorsolatearlBG_GPi.baseline = 1.25
DorsolatearlBG_Thal= Population(name="Thal_put", geometry = 2, neuron=LinearNeuron)
DorsolatearlBG_Thal.baseline = 0.85
Premotor=Population(name="Premotor", geometry = 2, neuron=LinearNeuron)


# ~~~~~~~~~~~~ DORSOMEDIAL LOOP (ACTION SELECTION) ~~~~~~~~~~~~
#shared with ventral loop
Visual_input = Population(name="Visual_input",geometry=(4,3), neuron=LinearNeuron)
Visual_input.baseline=0.0
Visual_input.noise=0.001

# acts as the Propio population in Scholl 2021
# PPTN in dorsomedial loop, previously called Realized_sequence
DorsomedialBG_PPTN = Population(name="DorsomedialBG_PPTN",geometry=8,neuron=InputNeuron)

#dorsal prefrontal cortex
#shared with dorsolateral loop
dPFC = Population(name="dPFC", geometry=6, neuron=LinearNeuron)
dPFC.tau = 40.0 # 40
dPFC.noise = 0.0

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

#previously called STN_caud
DorsomedialBG_STN = Population(name="DorsomedialBG_STN", geometry = (3), neuron=LinearNeuron)
DorsomedialBG_STN.tau = 10.0
DorsomedialBG_STN.noise = 0.01
DorsomedialBG_STN.baseline = 0.0

GPi = Population(name="GPi", geometry = 6, neuron=LinearNeuron)
GPi.tau = 10.0
GPi.noise = 0.5 #0.3
GPi.baseline = 1.9 #0.75 #1.5

GPe = Population(name="GPe", geometry = 6, neuron=LinearNeuron)
GPe.tau = 10.0
GPe.noise = 0.05
GPe.baseline = 1.0

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

#previously called SNc_caud
DorsomedialBG_SNc = Population(name='DorsomedialBG_SNc',geometry=8,neuron=DopamineNeuron)
DorsomedialBG_SNc.exc_threshold=0.2
DorsomedialBG_SNc.baseline = baseline_dopa_caud
DorsomedialBG_SNc.factor_inh = 1.0

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