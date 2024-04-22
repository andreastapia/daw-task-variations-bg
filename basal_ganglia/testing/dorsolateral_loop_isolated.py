from ANNarchy import (
    Population,
    Neuron,
    Projection,
    Monitor,
    enable_learning,
    simulate,
    compile
)
import random
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


#dorsal prefrontal cortex
#shared with dorsolateral loop
dPFC = Population(name="dPFC", geometry=6, neuron=LinearNeuron)
dPFC.tau = 40.0 # 40
dPFC.noise = 0.01

DorsolatearlBG_StriatumD1= Population(name="StrD1_put", geometry = 6, neuron=LinearNeuron)

DorsolatearlBG_GPi= Population(name="GPi_put", geometry = 2, neuron=LinearNeuron)
DorsolatearlBG_GPi.baseline = 0.75

DorsolatearlBG_Thal= Population(name="Thal_put", geometry = 2, neuron=LinearNeuron)
DorsolatearlBG_Thal.baseline = 0.85

Premotor=Population(name="Premotor", geometry = 2, neuron=LinearNeuron)

#TODO:fix names (enumerate connections)
# dorsolateral loop
# fixed dorsolateral loop connections
#dpfc to str_d1
dPFCStrD1_put=Projection(pre=dPFC,post=DorsolatearlBG_StriatumD1,target="exc")
dPFCStrD1_put.connect_one_to_one(weights=1.0)

#dpfc to dpfc
dPFCdPFC = Projection(pre=dPFC,post=dPFC,target='inh')
dPFCdPFC.connect_all_to_all(weights=0.2)

dPFCdPFC12 = Projection(pre=dPFC[0:2],post=dPFC[0:2],target='inh')
dPFCdPFC12.connect_all_to_all(weights=0.5)

dPFCdPFCAB = Projection(pre=dPFC[2:],post=dPFC[2:],target='inh')
dPFCdPFCAB.connect_all_to_all(weights=0.5)

#strd1 to gpi
StrD1GPi_put1=Projection(pre=DorsolatearlBG_StriatumD1[0:2],post=DorsolatearlBG_GPi,target="inh")
StrD1GPi_put1.connect_one_to_one(weights=1.0)

StrD1GPi_put2=Projection(pre=DorsolatearlBG_StriatumD1[2:4],post=DorsolatearlBG_GPi,target="inh")
StrD1GPi_put2.connect_one_to_one(weights=1.0)

StrD1GPi_put3=Projection(pre=DorsolatearlBG_StriatumD1[4:],post=DorsolatearlBG_GPi,target="inh")
StrD1GPi_put3.connect_one_to_one(weights=1.0)

#gpi to thal
GPiThal_put=Projection(pre=DorsolatearlBG_GPi,post=DorsolatearlBG_Thal,target="inh")
GPiThal_put.connect_one_to_one(weights=1.0)

#thal to premotor
ThalPM_put=Projection(pre=DorsolatearlBG_Thal,post=Premotor,target="exc")
ThalPM_put.connect_one_to_one(weights=1.0)

#premotor to premotor
PMPM = Projection(pre=Premotor,post=Premotor,target='inh')
PMPM.connect_all_to_all(weights=0.4) 

compile()

enable_learning()
num_trials = 400
eps=0.000001

gpi_activities=[]
dPFC_activities=[]
strd1_activities=[]

monitor_dPFC=Monitor(dPFC,"r")
monitor_dPFC.pause()

monitor_strd1=Monitor(DorsolatearlBG_StriatumD1,"r")
monitor_strd1.pause()

monitor_gpi=Monitor(DorsolatearlBG_GPi,"r")
monitor_gpi.pause()

left = 0
right = 0
for trial in range(num_trials):
    monitor_dPFC.resume()
    monitor_strd1.resume()
    monitor_gpi.resume()
    simulate(1000)
    monitor_dPFC.pause()
    monitor_strd1.pause()
    monitor_gpi.pause()

    dPFC[2].baseline=2.0

    simulate(100)

    dPFC[2].baseline=1.5

    # monitor_dPFC.resume()
    # monitor_strd1.resume()
    # monitor_gpi.resume()

    simulate(300)

    dPFC_activities.append(monitor_dPFC.get("r"))
    strd1_activities.append(monitor_strd1.get("r"))
    gpi_activities.append(monitor_gpi.get("r"))

    # monitor_dPFC.pause()
    # monitor_strd1.pause()
    # monitor_strd1.pause()

    motor_objectives=Premotor.r
    first_action=random.choices([0,1],weights=motor_objectives+eps,k=1)[0]
    
    if first_action ==0:
        left += 1
    
    if first_action ==1:
        right += 1


dpfc_act_np = np.array(dPFC_activities)
gpi_act_np = np.array(gpi_activities)
strd1_act_np = np.array(strd1_activities)


print("dpfc", dpfc_act_np.shape)
print("strd1", strd1_act_np.shape)
print("GPI", gpi_act_np.shape)

print("dpfc", np.average(dpfc_act_np[399], axis=0).shape)
print("strd1", np.average(strd1_act_np[399], axis=0).shape)
print("GPI", np.average(gpi_act_np[399], axis=0).shape)


print("dpfc", np.average(dpfc_act_np[399], axis=0))
print("strd1", np.average(strd1_act_np[399], axis=0))
print("GPI", np.average(gpi_act_np[399], axis=0))
print(left, right)