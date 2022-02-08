import power_flow_newton_raphson as pf
import numpy as np

##initialization
ybus = np.array([[complex(0,-10),complex(0,10)],[complex(0,10),complex(0,-10)]])
pset = np.array([2.0])
qset = np.array([1.0])

(n_buses, ymag, theta, jacobian, y, x) = pf.initialize_system(ybus, pset, qset)

#print(jacobian)
#print(x)
print(y)

#strange result... example is the one from lecture 7...