from ANNarchy import *
from ANNarchy.extensions.tensorboard import Logger
from hyperopt import fmin, tpe, hp, STATUS_OK
clear()
setup(dt=0.1)

COBA = Neuron(
    parameters="""
        El = -60.0          : population
        Vr = -60.0          : population
        Erev_exc = 0.0      : population
        Erev_inh = -80.0    : population
        Vt = -50.0          : population
        tau = 20.0          : population
        tau_exc = 5.0       : population
        tau_inh = 10.0      : population
        I = 20.0            : population
    """,
    equations="""
        tau * dv/dt = (El - v) + g_exc * (Erev_exc - v) + g_inh * (Erev_inh - v ) + I

        tau_exc * dg_exc/dt = - g_exc
        tau_inh * dg_inh/dt = - g_inh
    """,
    spike = "v > Vt",
    reset = "v = Vr",
    refractory = 5.0
)

P = Population(geometry=4000, neuron=COBA)
Pe = P[:3200]
Pi = P[3200:]
P.v = Normal(-55.0, 5.0)
P.g_exc = Normal(4.0, 1.5)
P.g_inh = Normal(20.0, 12.0)

Ce = Projection(pre=Pe, post=P, target='exc')
Ce.connect_fixed_probability(weights=0.6, probability=0.02)
Ci = Projection(pre=Pi, post=P, target='inh')
Ci.connect_fixed_probability(weights=6.7, probability=0.02)

compile()

m = Monitor(P, ['spike'])

simulate(1000.0)
data = m.get('spike')
fr = m.mean_fr(data)
print(fr)

logger = Logger()
def trial(args):
    
    # Retrieve the parameters
    w_exc = args[0]
    w_inh = args[1]
    
    # Reset the network
    reset()
    
    # Set the hyperparameters
    Ce.w = w_exc
    Ci.w = w_inh
    
    # Simulate 1 second
    simulate(1000.0)

    # Retrieve the spike recordings and the membrane potential
    spikes = m.get('spike')

    # Compute the population firing rate
    fr = m.mean_fr(spikes)
    print(fr)
    
    # Compute a quadratic loss around 30 Hz
    loss = 0.001*(fr - 30.0)**2   
    
    # Log the parameters
    logger.add_parameters({'w_exc': w_exc, 'w_inh': w_inh},
                         {'loss': loss, 'firing_rate': fr})
    
    return {
        'loss': loss,
        'status': STATUS_OK,
        # -- store other results like this
        'fr': fr,
        }

print(trial([0.6, 6.7]))

best = fmin(
    fn=trial,
    space=[
        hp.uniform('w_exc', 0.1, 1.0), 
        hp.uniform('w_inh', 1.0, 10.0)
    ],
    algo=tpe.suggest,
    max_evals=100)
print(best)

print(trial([best['w_exc'], best['w_inh']]))

logger.close()