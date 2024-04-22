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
dPFC = Population(name="dPFC", geometry=8, neuron=LinearNeuron)
dPFC.tau = 40.0 # 40
dPFC.noise = 0.01

Swap_input = Population(name="Swap_Input", geometry=8, neuron=LinearNeuron)
DorsolatearlBG_StriatumD1= Population(name="StrD1_put", geometry = 16, neuron=LinearNeuron)
#DorsolatearlBG_StriatumD1.baseline = -1.6

DorsolatearlBG_GPi= Population(name="GPi_put", geometry = 2, neuron=LinearNeuron)
DorsolatearlBG_GPi.baseline = 1.25

DorsolatearlBG_Thal= Population(name="Thal_put", geometry = 2, neuron=LinearNeuron)
DorsolatearlBG_Thal.baseline = 0.85

Premotor=Population(name="Premotor", geometry = 2, neuron=LinearNeuron)

# dorsolateral loop
# fixed dorsolateral loop connections
#dpfc to str_d1
dPFCStrD1_put1=Projection(pre=dPFC[0],post=DorsolatearlBG_StriatumD1[:2],target="exc")
dPFCStrD1_put1.connect_all_to_all(weights=1.0)

dPFCStrD1_put2=Projection(pre=dPFC[1],post=DorsolatearlBG_StriatumD1[2:4],target="exc")
dPFCStrD1_put2.connect_all_to_all(weights=1.0)

dPFCStrD1_put3=Projection(pre=dPFC[2],post=DorsolatearlBG_StriatumD1[4:6],target="exc")
dPFCStrD1_put3.connect_all_to_all(weights=1.0)

dPFCStrD1_put4=Projection(pre=dPFC[3],post=DorsolatearlBG_StriatumD1[6:8],target="exc")
dPFCStrD1_put4.connect_all_to_all(weights=1.0)

dPFCStrD1_put5=Projection(pre=dPFC[4],post=DorsolatearlBG_StriatumD1[8:10],target="exc")
dPFCStrD1_put5.connect_all_to_all(weights=1.0)

dPFCStrD1_put6=Projection(pre=dPFC[5],post=DorsolatearlBG_StriatumD1[10:12],target="exc")
dPFCStrD1_put6.connect_all_to_all(weights=1.0)

dPFCStrD1_put7=Projection(pre=dPFC[6],post=DorsolatearlBG_StriatumD1[12:14],target="exc")
dPFCStrD1_put7.connect_all_to_all(weights=1.0)

dPFCStrD1_put8=Projection(pre=dPFC[7],post=DorsolatearlBG_StriatumD1[14:],target="exc")
dPFCStrD1_put8.connect_all_to_all(weights=1.0)

#dpfc to dpfc
dPFCdPFC = Projection(pre=dPFC,post=dPFC,target='inh')
dPFCdPFC.connect_all_to_all(weights=0.2)

dPFCdPFC12 = Projection(pre=dPFC[0:4],post=dPFC[0:4],target='inh')
dPFCdPFC12.connect_all_to_all(weights=0.5)

dPFCdPFCAB = Projection(pre=dPFC[4:],post=dPFC[4:],target='inh')
dPFCdPFCAB.connect_all_to_all(weights=0.5)

#TODO:connect strd1 to strd1

#swap to str_d1
swapstrd1_put1 =Projection(pre=Swap_input[0],post=DorsolatearlBG_StriatumD1[:4:3],target="exc")
swapstrd1_put1.connect_all_to_all(weights=1.0)

swapstrd1_put2 =Projection(pre=Swap_input[1],post=DorsolatearlBG_StriatumD1[1:3],target="exc")
swapstrd1_put2.connect_all_to_all(weights=1.0)

swapstrd1_put3 =Projection(pre=Swap_input[2],post=DorsolatearlBG_StriatumD1[4:8:3],target="exc")
swapstrd1_put3.connect_all_to_all(weights=1.0)

swapstrd1_put4 =Projection(pre=Swap_input[3],post=DorsolatearlBG_StriatumD1[5:7],target="exc")
swapstrd1_put4.connect_all_to_all(weights=1.0)

swapstrd1_put5 =Projection(pre=Swap_input[4],post=DorsolatearlBG_StriatumD1[8:12:3],target="exc")
swapstrd1_put5.connect_all_to_all(weights=1.0)

swapstrd1_put6 =Projection(pre=Swap_input[5],post=DorsolatearlBG_StriatumD1[9:11],target="exc")
swapstrd1_put6.connect_all_to_all(weights=1.0)

swapstrd1_put7 =Projection(pre=Swap_input[6],post=DorsolatearlBG_StriatumD1[12:16:3],target="exc")
swapstrd1_put7.connect_all_to_all(weights=1.0)

swapstrd1_put8 =Projection(pre=Swap_input[7],post=DorsolatearlBG_StriatumD1[13:15],target="exc")
swapstrd1_put8.connect_all_to_all(weights=1.0)

# #swap to swap
# swapswap = Projection(pre=Swap_input, post=Swap_input, target='inh')
# swapswap.connect_all_to_all(weights=1.0)

#strd1 to gpi
StrD1GPi_put1=Projection(pre=DorsolatearlBG_StriatumD1[0:2],post=DorsolatearlBG_GPi,target="inh")
StrD1GPi_put1.connect_one_to_one(weights=1.0)

StrD1GPi_put2=Projection(pre=DorsolatearlBG_StriatumD1[2:4],post=DorsolatearlBG_GPi,target="inh")
StrD1GPi_put2.connect_one_to_one(weights=1.0)

StrD1GPi_put3=Projection(pre=DorsolatearlBG_StriatumD1[4:6],post=DorsolatearlBG_GPi,target="inh")
StrD1GPi_put3.connect_one_to_one(weights=1.0)

StrD1GPi_put4=Projection(pre=DorsolatearlBG_StriatumD1[6:8],post=DorsolatearlBG_GPi,target="inh")
StrD1GPi_put4.connect_one_to_one(weights=1.0)

StrD1GPi_put5=Projection(pre=DorsolatearlBG_StriatumD1[8:10],post=DorsolatearlBG_GPi,target="inh")
StrD1GPi_put5.connect_one_to_one(weights=1.0)

StrD1GPi_put6=Projection(pre=DorsolatearlBG_StriatumD1[10:12],post=DorsolatearlBG_GPi,target="inh")
StrD1GPi_put6.connect_one_to_one(weights=1.0)

StrD1GPi_put7=Projection(pre=DorsolatearlBG_StriatumD1[12:14],post=DorsolatearlBG_GPi,target="inh")
StrD1GPi_put7.connect_one_to_one(weights=1.0)

StrD1GPi_put8=Projection(pre=DorsolatearlBG_StriatumD1[14:],post=DorsolatearlBG_GPi,target="inh")
StrD1GPi_put8.connect_one_to_one(weights=1.0)

#gpi to thal
GPiThal_put=Projection(pre=DorsolatearlBG_GPi,post=DorsolatearlBG_Thal,target="inh")
GPiThal_put.connect_one_to_one(weights=1.0)

#thal to premotor
ThalPM_put=Projection(pre=DorsolatearlBG_Thal,post=Premotor,target="exc")
ThalPM_put.connect_one_to_one(weights=1.0)

#premotor to premotor
PMPM = Projection(pre=Premotor,post=Premotor,target='inh')
PMPM.connect_all_to_all(weights=0.4) 

StrD1StrD1_dorsolateral = Projection(pre=DorsolatearlBG_StriatumD1,post=DorsolatearlBG_StriatumD1,target='inh')
StrD1StrD1_dorsolateral.connect_all_to_all(weights=1.0) 

compile()

enable_learning()
num_trials = 10
eps=0.000001

gpi_activities=[]
dPFC_activities=[]
strd1_activities=[]
swap_activities=[]

monitor_dPFC=Monitor(dPFC,"r")
monitor_dPFC.pause()

monitor_strd1=Monitor(DorsolatearlBG_StriatumD1,"r")
monitor_strd1.pause()

monitor_gpi=Monitor(DorsolatearlBG_GPi,"r")
monitor_gpi.pause()

monitor_swap=Monitor(Swap_input,"r")
monitor_swap.pause()

left = 0
right = 0
for trial in range(num_trials):
    Swap_input.baseline=0

    monitor_dPFC.resume()
    monitor_strd1.resume()
    monitor_gpi.resume()
    monitor_swap.resume()
    simulate(1000)
    monitor_dPFC.pause()
    monitor_strd1.pause()
    monitor_gpi.pause()
    monitor_swap.pause()

    #DC
    Swap_input[7].baseline=1.5
    #E
    dPFC[6].baseline=2.0

    simulate(100)

    Swap_input[7].baseline=1.5
    dPFC[6].baseline=1.5

    # monitor_dPFC.resume()
    # monitor_strd1.resume()
    # monitor_gpi.resume()
    # monitor_swap.resume()

    simulate(300)

    dPFC_activities.append(monitor_dPFC.get("r"))
    strd1_activities.append(monitor_strd1.get("r"))
    gpi_activities.append(monitor_gpi.get("r"))
    swap_activities.append(monitor_swap.get("r"))

    # monitor_dPFC.pause()
    # monitor_strd1.pause()
    # monitor_strd1.pause()
    # monitor_swap.pause()

    motor_objectives=Premotor.r
    first_action=random.choices([0,1],weights=motor_objectives+eps,k=1)[0]
    
    if first_action ==0:
        left += 1
    
    if first_action ==1:
        right += 1


swap_act_np = np.array(swap_activities)
dpfc_act_np = np.array(dPFC_activities)
gpi_act_np = np.array(gpi_activities)
strd1_act_np = np.array(strd1_activities)


print("SWAP", swap_act_np.shape)
print("dpfc", dpfc_act_np.shape)
print("strd1", strd1_act_np.shape)
print("GPI", gpi_act_np.shape)

print("SWAP", np.average(swap_act_np[9], axis=0).shape)
print("dpfc", np.average(dpfc_act_np[9], axis=0).shape)
print("strd1", np.average(strd1_act_np[9], axis=0).shape)
print("GPI", np.average(gpi_act_np[9], axis=0).shape)


print("SWAP", np.average(swap_act_np[9], axis=0))
print("dpfc", np.average(dpfc_act_np[9], axis=0))
print("strd1", np.average(strd1_act_np[9], axis=0))
print("GPI", np.average(gpi_act_np[9], axis=0))
print(left, right)
