from ANNarchy import Projection, Normal, Uniform, compile
from .params import *
from .populations import *
from .synapses import *

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

# PROJECTIONS FOR VENTRAL LOOP
for i in range(0,num_stim):
    
    #Input from the stimulus representing cells to the striatum, main plastic connections to the loop.
    #The input goes to the 2 striatal populations
    #create a projection for every column of neurons from hipocampus to NAccD1
    #the store it in an array
    ITStrD1_shell.append(Projection(pre=Hippocampus,post=NAccD1_shell[:,i],target='exc',synapse=DAPostCovarianceNoThreshold))
    ITStrD1_shell[i].connect_all_to_all(weights = Normal(0.1/(num_stim/2),0.02)) # perhaps also scale by numer of goals #0.02
    ITStrD1_shell[i].tau = 100 #100
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
ThalThal.connect_all_to_all(weights=0.2) # was 1.1 # perhaps needs to be lower

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
StrD1SNr_core.connect_one_to_one(weights=1.0)

SNrThal_core = Projection(pre=WM_SNr_core,post=VentralBG_Thal,target='inh')
SNrThal_core.connect_one_to_one(weights=0.75)

ObjStrThal_shell= Projection(pre=mPFC,post=VentralBG_Cortical_feedback,target='exc')
ObjStrThal_shell.connect_one_to_one(weights=1.2)

#~~~~~~~ DORSOMEDIAL ~~~~~~~~
#dorsomedial loop
ObjStrD1_caud = Projection(pre=mPFC,post=DorsomedialBG_StrD1,target='exc',synapse=DAPostCovarianceNoThreshold_trace) 
ObjStrD1_caud.connect_all_to_all(weights = Normal(0.5,0.2))
ObjStrD1_caud.tau = 400 #100
ObjStrD1_caud.regularization_threshold = 2.0
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
StrD1StrD1_caud=Projection(pre=DorsomedialBG_StrD1,post=DorsomedialBG_StrD1,target='inh')
StrD1StrD1_caud.connect_all_to_all(weights = 0.2)

StrD2StrD2_caud=Projection(pre=DorsomedialBG_StrD2,post=DorsomedialBG_StrD2,target='inh')
StrD2StrD2_caud.connect_all_to_all(weights= weight_inh_sd2/num_stim)

StrThalStrThal_caud = Projection(pre=DorsomedialBG_cortical_feedback,post=DorsomedialBG_cortical_feedback,target='inh')
StrThalStrThal_caud.connect_all_to_all(weights=weight_inh_thal/num_stim) #0.5

# STNSTN_dorsomedial = Projection(pre=DorsomedialBG_STN, post=DorsomedialBG_STN, target='inh')
# STNSTN_dorsomedial.connect_all_to_all(weights=0.0)

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

#~~~~~~~ DORSOLATERAL ~~~~~~~~
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