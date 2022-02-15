""" import power_flow_newton_raphson as pf
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
 """

import power_flow_newton_raphson as pf
import numpy as np

##Working example: Vittal example 10.6
#Realized that it was probably wiser to program using the vectors 
#v, delta, d_p, d_q and stack them when necessary for calculations

##initialization
slack_bus_idx = 0 #slack bus placement - bus 1 is bus 0 in the code
pv_idx = np.array([])

iteration_limit = 15
ybus = np.array([[complex(0,-10),complex(0,10)],[complex(0,10),complex(0,-10)]])
#power and voltage setpoints
pset = np.array([-2.0]) #P setpoints for every bus except slack bus in ascending bus order
qset = np.array([-1.0]) #Q setpoints for every bus except slack bus and PV buses in ascending bus order
qset.shape = (np.size(qset),1)
pset.shape = (np.size(pset),1)

#setup system variables
(jacobian, n_buses, vmag, delta, vmag_full, delta_full, g, b, p, q, del_p, del_q) = pf.initialize_system(ybus, pset, qset, pv_idx, 0)

#noting indices of PQ-busses based on slack bus and PV-bus indices  
pq_idx = np.arange(n_buses, dtype=int)
for val in pv_idx:
    pq_idx = pq_idx[pq_idx != val]
pq_idx = pq_idx[(pq_idx != slack_bus_idx)]

#Calculating initial power vectors
(p, q, p_full, q_full) = pf.calculate_power_vecs(n_buses, vmag_full, delta_full, b, g, pv_idx)
#Updating mismatch vector
(del_p, del_q) = pf.update_mismatch_vector(p, q, pset, qset)

#Calculating Jacobian matrix
pf.calculate_jacobian(n_buses, jacobian, vmag_full, delta_full, g, b, p_full, q_full)

#simplify Jacobian according to PV-busses
jacobian_calc = pf.simplify_jacobian(n_buses, pv_idx, jacobian)
print("J: \n", np.round(jacobian_calc, 2))


""" #calculate next iteration of voltage magnitude and phase angle
(delta, vmag) = pf.next_iteration(jacobian_calc, vmag, delta, del_p, del_q)
print("delta_next:\n", delta * 180/np.pi)
print("vmag_next:\n", vmag) """


for i in range(1, iteration_limit + 1):
    (delta, vmag) = pf.next_iteration(jacobian_calc, vmag, delta, del_p, del_q)
    #Calculating initial power vectors
    delta_full[1:] = delta
    vmag_full[1] = vmag #PLACEHOLDER - NEEDS LOGIC FOR OTHER SIZES!!!
    (p, q, p_full, q_full) = pf.calculate_power_vecs(n_buses, vmag_full, delta_full, b, g, pv_idx)
    (del_p, del_q) = pf.update_mismatch_vector(p, q, pset, qset)
    y = np.row_stack((del_p, del_q))
    if pf.check_convergence(y, 0.001):
        print("Power flow converged at %d iterations.\n" % i)
        print("delta:\n",delta * 180/np.pi)
        print("vmag:\n",vmag)
        print("mismatch vector:\n", y)
        break
    
    elif i == iteration_limit:
        print("Power flow did not converge after %d iterations.\n" % i )


#SEMI-WORKING... A bit strange result and lots of iterations. Will revisit.
