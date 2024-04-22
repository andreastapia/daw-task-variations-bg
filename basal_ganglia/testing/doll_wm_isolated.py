from ANNarchy import (
    Population,
    Neuron,
    Projection,
    Monitor,
    enable_learning,
    simulate,
    compile
)
import numpy as np

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


# ~~~~~~~~~~~~ WORKING MEMORY ~~~~~~~~~~~~
#previously called StrD1_core
#shared with dorsomedial loop
#num_objectives is the same in doll task (2015)
num_objectives=8
# Neurons for the ventral loop, previously called "Objectives"
#mPFC, is shared with working memory and ventral loop
mPFC = Population(name="mPFC_Objectives", geometry=num_objectives, neuron=LinearNeuron)
mPFC.tau = 40.0 # 40
mPFC.noise = 0.01

# Thal for WM/ventral loop, previously called Thal_vent
VentralBG_Thal = Population(name="VentralBG_Thal", geometry=num_objectives, neuron=LinearNeuron)
VentralBG_Thal.tau = 10.0
VentralBG_Thal.noise = 0.025 / 2.0
VentralBG_Thal.baseline = 1.2

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

## handcrafted working memory
ObjStrD1_core = Projection(pre=mPFC,post=WM_StrD1_core,target='exc')
ObjStrD1_core.connect_one_to_one(weights=2.0)    

StrD1SNr_core = Projection(pre=WM_StrD1_core,post=WM_SNr_core,target='inh')
StrD1SNr_core.connect_one_to_one(weights=1.0)

SNrThal_core = Projection(pre=WM_SNr_core,post=VentralBG_Thal,target='inh')
SNrThal_core.connect_one_to_one(weights=0.75)

ThalObj = Projection(pre=VentralBG_Thal,post=mPFC,target='exc')
ThalObj.connect_one_to_one(weights=1.0) #3.0

compile()

mpfc_activities=[]
thal_activities=[]
strd1core_activities=[]
snr_activities=[]

monitor_mpfc=Monitor(mPFC,'r')
monitor_mpfc.pause()

monitor_thal=Monitor(VentralBG_Thal,'r')
monitor_thal.pause()

monitor_strd1=Monitor(WM_StrD1_core,'r')
monitor_strd1.pause()

monitor_snr=Monitor(WM_SNr_core,'r')
monitor_snr.pause()

num_trials = 401
for trial in range(num_trials):
    enable_learning()
    #reset everything at start of trial
    #ventral loop
    # mPFC.r = 0.0
    # mPFC.baseline = 0.0
    # VentralBG_Thal.r=0
    # VentralBG_Thal.baseline=0

    simulate(1000)
    VentralBG_Thal[0].baseline=2.0
    VentralBG_Thal[1:].baseline=0
    simulate(200)  
    monitor_mpfc.resume()
    monitor_thal.resume()
    monitor_strd1.resume()
    monitor_snr.resume()
    if trial > 0:
        mpfc_activities.append(monitor_mpfc.get("r"))
        thal_activities.append(monitor_thal.get("r"))
        strd1core_activities.append(monitor_strd1.get("r"))
        snr_activities.append(monitor_snr.get("r"))

mpfc_act_np = np.array(mpfc_activities)
thal_act_np = np.array(thal_activities)
strd1core_act_np = np.array(strd1core_activities)
snr_act_np = np.array(snr_activities)

print("mpfc", mpfc_act_np.shape)
print("strd1core", strd1core_act_np.shape)
print("snr", snr_act_np.shape)
print("thal", thal_act_np.shape)

print("mpfc", np.average(mpfc_act_np[399], axis=0).shape)
print("strd1core", np.average(strd1core_act_np[399], axis=0).shape)
print("snr", np.average(snr_act_np[399], axis=0).shape)
print("thal", np.average(thal_act_np[399], axis=0).shape)

print("mpfc", np.average(mpfc_act_np[399], axis=0))
print("strd1core", np.average(strd1core_act_np[399], axis=0))
print("snr", np.average(snr_act_np[399], axis=0))
print("thal", np.average(thal_act_np[399], axis=0))