#General networks parameters
#Populations setups
#same objectives amount as daw task (2011)
num_objectives=8
baseline_dopa_vent = 0.15
gen_tau_trace=100

#shared between populations and projections
baseline_dopa_caud=0.1
#8 stimulus (2 face, 2 body parts, 2 tools, 2 scenes)
num_stim = 8

#Projections exclusive
# caudate
weight_local_inh = 0.8 *0.2 * 0.166 #old model (Scholl) with 6 distinct population -> 0.8*0.2, here only one -> scale it by 1/6
weight_stn_inh = 0.3 
weight_inh_sd2 = 0.5 *0.2
weight_inh_thal = 0.5 *0.2
modul = 0.01

