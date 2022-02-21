import power_flow_newton_raphson as pf
import numpy as np
import pandapower as pp
import pandapower.networks as nw


# To be written: Tests comparing custom solver results to PandaPower results using their test cases

# Will probably need to implement a function to check reactive power limits for PV busses

# Will also need to add the ability for loads to be attached to PV-busses. 
# The way to do this would probably be to handle generator loads as offsets. See the Teams discussion.

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

for i in range(len(gen.index)):
    gen_list.append({'type':'pv', 'bus':gen.bus[i], 'vset':gen.vm_pu[i], 'pset':gen.p_mw[i], 'qset':None, 'qmin':gen.min_q_mvar[i], 'qmax':gen.max_q_mvar[i], 'pmin':gen.min_p_mw[i], 'pmax':gen.max_p_mw[i]})

for i in range(len(load.index)):
    load_list.append({'bus':load.bus[i], 'p':load.p_mw[i], 'q':load.q_mvar[i]})

system.update({'generators':gen_list})
system.update({'loads':load_list})

#Consider the possibility of a load connected to the same bus as the slack generator...
#This is the case for case4gs

#%%

(n_buses, g, b) = pf.process_admittance_mat(system)

(vmag, delta, vmag_full, delta_full) = pf.init_voltage_vecs(system)

(p, q, p_full, q_full) = pf.calc_power_vecs(system, vmag_full, delta_full, g, b)

#!!!!!!!!!!!!!!!!!!!!
#Not converging... Need to finish calc_power_vecs such that loads on generator bussed are considered!


#%%

# jacobian = pf.calc_jacobian(system, vmag_full, delta_full, g, b, p_full, q_full)

# jacobian_calc = pf.jacobian_calc_simplify(system, jacobian)

# (pset, qset) = pf.calc_power_setpoints(system)

# (del_p, del_q) = calc_mismatch_vecs(system, p, q)

# pv_idx = get_pv_idx(system)
# non_pv_idx = np.arange(n_buses)
# non_pv_idx = np.delete(non_pv_idx, pv_idx, 0)
# iteration_limit = system.get('iteration_limit')
# tolerance = system.get('tolerance')

# for i in range(1, iteration_limit + 1):
#     (delta, vmag) = next_iteration(jacobian_calc, vmag, delta, del_p, del_q)
#     #Calculating initial power vectors
    
#     delta_full[1:] = delta #updating voltage angles on all busses except slack
#     vmag_full[non_pv_idx[1:]] = vmag #updating voltage magnitudes on non-slack and non-PV busses
    
#     (p, q, p_full, q_full) = calc_power_vecs(system, vmag_full, delta_full, g, b)

#     jacobian = calc_jacobian(system, vmag_full, delta_full, g, b, p_full, q_full)

#     jacobian_calc = jacobian_calc_simplify(system, jacobian)

#     (del_p, del_q) = calc_mismatch_vecs(system, p, q)
    
#     y = np.row_stack((del_p, del_q))


#     # print("\nIteration %d:\n" % i)
#     # print("delta:\n",delta * 180/np.pi)
#     # print("vmag:\n",vmag)
#     # print("mismatch vector:\n", y)
#     # print("Jacobian:\n", jacobian_calc)

#     if check_convergence(y, tolerance):
#         print("Power flow converged at %d iterations.\n" % i)
#         print("Phase angles (unknowns):\n",delta * 180/np.pi)
#         print("Voltage magnitudes (unknowns):\n",vmag)
#         print("Real power (all buses, injections):\n", p_full)
#         print("Reactive power (all buses, injections):\n", q_full)
#         print("Mismatch vector for known injections:\n", y)
#         break
    
#     elif i == iteration_limit:
#         print("Power flow did not converge after %d iterations.\n" % i )

#%%

#pf.run_newton_raphson(system)