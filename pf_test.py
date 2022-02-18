import power_flow_newton_raphson as pf
import numpy as np
import pandapower as pp
import pandapower.networks as nw


# To be written: Tests comparing custom solver results to PandaPower results using their test cases

# Will probably need to implement a function to check reactive power limits for PV busses

# Will also need to add the ability for loads to be attached to PV-busses. 
# The way to do this would probably be to handle generator loads as offsets. See the Teams discussion.

###################################################################################################



######MAIN IDEA FOR IMPROVEMENT FOR ORGANIZING SYSTEM DATA:#######

#Using dictionaries!
#System dictionary containing: 
    #List of generator, load etc. dictionaries
    #   Each of these contain parameters such as placement (bus), setpoints and limits
    #parameters such as tolerance, max iterations

###################################################################################################


baseMVA = 100.0 #base power for system

network = nw.case4gs()

pp.runpp(network, enforce_q_lims=False) #run power flow
ybus = network._ppc["internal"]["Ybus"].todense() #extract Ybus after running power flow
gen = network.gen
load = network.load

#%% View results of PandaPower powerflow
pf_results = network.res_bus
pf_results['p_pu'] = pf_results.p_mw/baseMVA
pf_results['q_pu'] = pf_results.q_mvar/baseMVA
print(pf_results)


#%% Custom code comparison
##initialization

system = {'admmat':ybus,'slack_idx':0,'iteration_limit':15,'tolerance':0.001,'generators':[],'loads':[]}

gen_list = []
load_list = []

#Fill lists of generator and load dictionaries based on the loaded generator and load information from PandaPower


#Rewrite functions to accept system dictionary as input
#Consider an additional dictionary or expansion of the system dictionary to store some of the information stored as vectors in earlier code




#%%




#old code:
""" 
pv_idx = np.array([1])

#power and voltage setpoints
pset = np.array([0.6661, -2.8653]) #P setpoints for every bus except slack bus in ascending bus order
qset = np.array([-1.2244]) #Q setpoints for every bus except slack bus and PV buses in ascending bus order
vset = np.array([1.05]) #voltage setpoints for every PV bus in ascending bus order
vset.shape = (np.size(vset),1)
qset.shape = (np.size(qset),1)
pset.shape = (np.size(pset),1)

#setup system variables
(jacobian, n_buses, vmag, delta, vmag_full, delta_full, g, b, p, q, del_p, del_q) = pf.initialize_system(ybus, pset, qset, pv_idx, vset)

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


for i in range(1, iteration_limit + 1):
    (delta, vmag) = pf.next_iteration(jacobian_calc, vmag, delta, del_p, del_q)
    #Calculating initial power vectors
    delta_full[1:] = delta
    vmag_full[2] = vmag #PLACEHOLDER - NEEDS LOGIC FOR OTHER SIZES!!!
    (p, q, p_full, q_full) = pf.calculate_power_vecs(n_buses, vmag_full, delta_full, b, g, pv_idx)

    pf.calculate_jacobian(n_buses, jacobian, vmag_full, delta_full, g, b, p_full, q_full)
    #simplify Jacobian according to PV-busses
    jacobian_calc = pf.simplify_jacobian(n_buses, pv_idx, jacobian)

    (del_p, del_q) = pf.update_mismatch_vector(p, q, pset, qset)
    y = np.row_stack((del_p, del_q))

    print("\nIteration %d:\n" % i)
    print("delta:\n",delta * 180/np.pi)
    print("vmag:\n",vmag)
    print("mismatch vector:\n", y)
    print("Jacobian:\n", jacobian_calc)

    if pf.check_convergence(y, tolerance):
        print("Power flow converged at %d iterations.\n" % i)
        print("delta:\n",delta * 180/np.pi)
        print("vmag:\n",vmag)
        print("Real power flows excluding slack:\n", p_full[1:])
        print("Reactive power flows excluding slack:\n", q_full[1:])
        print("mismatch vector:\n", y)
        break
    
    elif i == iteration_limit:
        print("Power flow did not converge after %d iterations.\n" % i )
 """