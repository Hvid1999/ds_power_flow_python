import power_flow_newton_raphson as pf
import numpy as np

##initialization
iteration_limit = 15
ybus = np.array([[complex(0,-10),complex(0,10)],[complex(0,10),complex(0,-10)]])
n_buses = ybus.shape[0]
#buses = [pf.Bus() for i in range(n_buses)]
#buses[0].type = 'slack'

pset = np.array([0,-2.0])
qset = np.array([0,-1.0])

(jacobian, y, x, g, b, p, q) = pf.initialize_system(n_buses, ybus, pset, qset)
print("\nInitial values:\n")
print("x:\n",x)
print("J: \n", np.round(jacobian, 2))
print("y:\n", y)

#The test script calculates the initial parameters for the example in the lecture 7 slides
#next step is to implement the iterative solution

for i in range(1, iteration_limit + 1):
    pf.next_iteration(jacobian, x, y, n_buses, b, g, p, q, pset, qset)
    if pf.check_convergence(y, 0.001):
        print("Power flow converged at %d iterations.\n" % i)
        print("x:\n",x)
        print("y:\n", y)
        break
    if i == iteration_limit:
        print("Power flow did not converge after %d iterations.\n" % i )
