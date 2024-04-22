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
import matplotlib.pyplot as plt
from hyperopt import fmin, tpe, hp, STATUS_OK

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
num_objectives=8
# Neurons for the ventral loop, previously called "Objectives"
#mPFC, is shared with working memory and ventral loop
mPFC = Population(name="mPFC_Objectives", geometry=num_objectives, neuron=LinearNeuron)
mPFC.tau = 40.0 # 40
mPFC.noise = 0.01

# Thal for WM/ventral loop, previously called Thal_vent
VentralBG_Thal = Population(name="VentralBG_Thal", geometry=num_objectives, neuron=LinearNeuron)
VentralBG_Thal.tau = 10.0
VentralBG_Thal.noise = 0.0125
VentralBG_Thal.baseline = 1.2

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

## handcrafted working memory
ThalObj = Projection(pre=VentralBG_Thal,post=mPFC,target='exc')
ThalObj.connect_one_to_one(weights=1.0) #3.0

ObjStrD1_core = Projection(pre=mPFC,post=WM_StrD1_core,target='exc')
ObjStrD1_core.connect_one_to_one(weights=2.0)    

StrD1SNr_core = Projection(pre=WM_StrD1_core,post=WM_SNr_core,target='inh')
StrD1SNr_core.connect_one_to_one(weights=1.0)

SNrThal_core = Projection(pre=WM_SNr_core,post=VentralBG_Thal,target='inh')
SNrThal_core.connect_one_to_one(weights=0.75)

ObjObj = Projection(pre=mPFC,post=mPFC,target='inh')
ObjObj.connect_all_to_all(weights=0.8) 

ThalThal = Projection(pre=VentralBG_Thal,post=VentralBG_Thal,target='inh')
ThalThal.connect_all_to_all(weights=0.1) # was 1.1 # perhaps needs to be lower


compile()

enable_learning()

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

num_trials = 400

def trial(args):

    sequence = 4
    w_thal_mpfc = args[0]
    w_mpfc_strd1 = args[1]
    w_strd1_snr = args[2]
    w_snr_thal = args[3]
    w_mpfc_mpfc = args[4]
    w_thal_thal = args[5]

    ThalObj.w = w_thal_mpfc
    ObjStrD1_core.w = w_mpfc_strd1
    StrD1SNr_core.w = w_strd1_snr
    SNrThal_core.w = w_snr_thal
    ObjObj.w = w_mpfc_mpfc
    ThalThal.w = w_thal_thal

    mPFC.r = 0.0
    mPFC.baseline = 0.0
    VentralBG_Thal.r=0
    VentralBG_Thal.baseline=0

    simulate(2000)
 
    VentralBG_Thal.baseline=1.2  
    
    monitor_mpfc.resume()
    monitor_thal.resume()
    monitor_strd1.resume()
    monitor_snr.resume()

    simulate(200)  

    mPFC[sequence].baseline=2.0
    simulate(100)
    mPFC[sequence].baseline=0.2
    simulate(300)

    monitor_mpfc.pause()
    monitor_thal.pause()
    monitor_strd1.pause()
    monitor_snr.pause()

    mpfc_rates = monitor_mpfc.get("r")

    mpfc_rates_np = np.array(mpfc_rates)
    mean_rates_avg = np.average(mpfc_rates_np, axis=0)

    loss = (1/600)*(mean_rates_avg[sequence] - 1.0)**2

    mpfc_activities.append(mpfc_rates)

    return {
        'loss': loss,
        'status': STATUS_OK,
        # -- store other results like this
        'mean_act': mean_rates_avg[sequence],
        }

#trial([1.0,2.0,1.0,0.75,0.8,0.1])

#trial([1.62,3.75,2.07,3.01,0.98,4.53])

#trial([1.74, 1.32, 0.73, 1.92, 1.67, 0.26])

trial([0.93, 0.68, 1.30, 1.62, 1.26, 0.83])

# best = fmin(
#     fn=trial,
#     space=[
#         hp.uniform('w_thal_mpfc', 0.0, 2.0), 
#         hp.uniform('w_mpfc_strd1', 0.0, 2.0),
#         hp.uniform('w_strd1_snr', 0.0, 2.0),
#         hp.uniform('w_snr_thal', 0.0, 2.0),
#         hp.uniform('w_mpfc_mpfc', 0.0, 2.0),
#         hp.uniform('w_thal_thal', 0.0, 2.0)
#     ],
#     algo=tpe.suggest,
#     max_evals=100)

# print([best['w_thal_mpfc'], best['w_mpfc_strd1'], best['w_strd1_snr'], best['w_snr_thal'], best['w_mpfc_mpfc'], best['w_thal_thal']])

# print(trial([best['w_thal_mpfc'], best['w_mpfc_strd1'], best['w_strd1_snr'], best['w_snr_thal'], best['w_mpfc_mpfc'], best['w_thal_thal']]))


mpfc_act_np = np.array(mpfc_activities)
print("mpfc", mpfc_act_np.shape)
print("mpfc", np.average(mpfc_act_np, axis=1).shape)
print("mpfc", np.average(mpfc_act_np, axis=1))


fig, (ax1) = plt.subplots(1)
fig.tight_layout()
ax1.plot(np.arange(mpfc_act_np.shape[1]), mpfc_act_np[0][:,0], label='(1,A)')
ax1.plot(np.arange(mpfc_act_np.shape[1]), mpfc_act_np[0][:,1], label='(1,B)')
ax1.plot(np.arange(mpfc_act_np.shape[1]), mpfc_act_np[0][:,2], label='(1,C)')
ax1.plot(np.arange(mpfc_act_np.shape[1]), mpfc_act_np[0][:,3], label='(1,D)')
ax1.plot(np.arange(mpfc_act_np.shape[1]), mpfc_act_np[0][:,4], label='(2,A)')
ax1.plot(np.arange(mpfc_act_np.shape[1]), mpfc_act_np[0][:,5], label='(2,B)')
ax1.plot(np.arange(mpfc_act_np.shape[1]), mpfc_act_np[0][:,6], label='(2,C)')
ax1.plot(np.arange(mpfc_act_np.shape[1]), mpfc_act_np[0][:,7], label='(2,D)')
ax1.legend(loc='upper right', title='Sequence')
plt.show()
