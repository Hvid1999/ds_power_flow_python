import power_flow_newton_raphson as pf
import numpy as np

##Working example: Vittal example 10.6

##initialization
iteration_limit = 15
ybus = np.array([[complex(0,-19.98),complex(0,10),complex(0,10)],[complex(0,10),complex(0,-19.98),complex(0,10)],[complex(0,10),complex(0,10),complex(0,-19.98)]])
n_buses = ybus.shape[0]
pset = np.array([0, 0.6661, 2.8653])
qset = np.array([0, 0, 1.2244])
vset = np.array([1.0, 1.05, 1.0])
vset.shape = (3,1)
qset.shape = (3,1)
pset.shape = (3,1)


buses = [pf.Bus() for i in range(n_buses)]
buses[0].type = 'slack'

for i in range(0, n_buses):
    if vset[i] != 1.0:
        buses[i].type = 'pv'
        buses[i].gen = True
    print("Bus %d " % (i+1), buses[i])


#not working properly yet
#(jacobian, y, x, g, b, p, q) = pf.initialize_system(n_buses, ybus, pset, qset, True, vset)

g = np.real(ybus)
b = np.imag(ybus)
delta = np.zeros((n_buses,1))
vmag = vset
x = np.row_stack((delta,vset))

#Setup initial power mismatch vector delta_y
p = np.zeros((n_buses,1))
q = np.zeros((n_buses,1))
y = np.zeros((n_buses,1))
pf.calculate_power_vecs(n_buses, vmag, delta, b, g, p, q)
pf.update_mismatch_vector(y, n_buses, p, q, pset, qset)

#Setup initial jacobian
#jacobian = np.zeros((2*(n_buses-1),2*(n_buses-1)))
#pf.calculate_jacobian(n_buses, jacobian, vmag, delta, g, b, p, q)

""" 
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
 """