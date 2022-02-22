import power_flow_newton_raphson as pf
import numpy as np

##Working example: Vittal example 10.6
#Realized that it was probably wiser to program using the vectors 
#v, delta, d_p, d_q and stack them when necessary for calculations


#bus admittance matrix
ybus = np.array([[complex(0,-19.98),complex(0,10),complex(0,10)],[complex(0,10),complex(0,-19.98),complex(0,10)],[complex(0,10),complex(0,10),complex(0,-19.98)]])

slack_dict = {'bus':0, 'vset':1.0, 'pmin':None, 'pmax':None, 'qmin':None, 'qmax':None}

system = {'admmat':ybus,'slack':slack_dict,'iteration_limit':15,'tolerance':1e-3,'generators':[],'loads':[]}

#manually entering generators and loads in this example
gen_list = [{'type':'pv', 'bus':1, 'vset':1.05, 'pset':0.6661, 'qset':None, 'qmin':None, 'qmax':None, 'pmin':None, 'pmax':None}]
load_list = [{'bus':2, 'p':2.8653, 'q':1.2244}]

system.update({'generators':gen_list})
system.update({'loads':load_list})

pf.run_newton_raphson(system)
