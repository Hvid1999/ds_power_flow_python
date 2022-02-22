import power_flow_newton_raphson as pf
import pandapower as pp
import pandapower.networks as nw
import numpy as np
import pandas as pd
###################################################################################################

#Cases with slack bus on other index than 1 (0)
#case5
#case24_ieee_rts
#case39
#...


network = nw.case39()

#Function loading system information from PandaPower network
#system = pf.load_pandapower_case(network)

e_q_lims = True

baseMVA = network.sn_mva #base power for system

pp.runpp(network, enforce_q_lims=e_q_lims) #run power flow
ybus = network._ppc["internal"]["Ybus"].todense() #extract Ybus after running power flow
gen = network.gen
load = network.load
slack = network.ext_grid

#Saving PandaPower results and per-unitizing power values
pf_results = network.res_bus
pf_results['p_pu'] = pf_results.p_mw/baseMVA
pf_results['q_pu'] = pf_results.q_mvar/baseMVA
pf_results = pf_results[['vm_pu','va_degree','p_pu','q_pu']]

slack_dict = {'bus':slack.bus[0], 'vset':slack.vm_pu[0], 'pmin':slack.min_p_mw[0]/baseMVA,
              'pmax':slack.max_p_mw[0]/baseMVA, 'qmin':slack.min_q_mvar[0]/baseMVA, 
              'qmax':slack.max_q_mvar[0]/baseMVA}
##Setup system dictionary
system = {'admmat':ybus,'slack':slack_dict,'generators':[],'loads':[],
          'iteration_limit':15,'tolerance':1e-3}

gen_list = []
load_list = []

#Fill lists of generator and load dictionaries based on the loaded generator and load information from PandaPower
#Per-unitizing the values according to the power base

for i in range(len(gen.index)):
    gen_list.append({'type':'pv', 'bus':gen.bus[i], 'vset':gen.vm_pu[i], 'pset':gen.p_mw[i]/baseMVA,
                     'qset':None, 'qmin':gen.min_q_mvar[i]/baseMVA, 'qmax':gen.max_q_mvar[i]/baseMVA,
                     'pmin':gen.min_p_mw[i]/baseMVA, 'pmax':gen.max_p_mw[i]/baseMVA})

for i in range(len(load.index)):
    load_list.append({'bus':load.bus[i], 'p':load.p_mw[i]/baseMVA, 'q':load.q_mvar[i]/baseMVA})

system.update({'generators':gen_list})
system.update({'loads':load_list})

#%%
pf.run_newton_raphson(system, enforce_q_limits=e_q_lims)
print('\nPandaPower results:\n')
print(pf_results)

#%%

# (n_buses, g, b) = pf.process_admittance_mat(system)
# q_loads = np.zeros((n_buses,1))
# for l in system.get('loads'):
#     k = l.get('bus')
#     q_loads[k] = -l.get('q')


#%%

# (n_buses, g, b) = pf.process_admittance_mat(system)


# (vmag, delta, vmag_full, delta_full) = pf.init_voltage_vecs(system)

# (p, q, p_full, q_full) = pf.calc_power_vecs(system, vmag_full, delta_full, g, b)

# jacobian = pf.calc_jacobian(system, vmag_full, delta_full, g, b, p_full, q_full)

# jacobian_calc = pf.jacobian_calc_simplify(system, jacobian)

# (pset, qset) = pf.calc_power_setpoints(system)

# (del_p, del_q) = pf.calc_mismatch_vecs(system, p, q)

# #obtaining list of non-PV and non-slack busses
# pv_idx = pf.get_pv_idx(system)
# pq_idx = np.arange(n_buses)
# non_slack_idx = np.delete(pq_idx, pf.slack_idx(system), 0)
# pq_idx = np.delete(pq_idx, pv_idx, 0)
# pq_idx = pq_idx[pq_idx != pf.slack_idx(system)]

# iteration_limit = system.get('iteration_limit')
# tolerance = system.get('tolerance')



#%%
# for i in range(1, iteration_limit + 1):
#     (delta, vmag) = pf.next_iteration(jacobian_calc, vmag, delta, del_p, del_q)
#     #Calculating initial power vectors
    
#     delta_full[non_slack_idx] = delta #updating voltage angles on all busses except slack
#     vmag_full[pq_idx] = vmag #updating voltage magnitudes on non-slack and non-PV busses
    
#     (p, q, p_full, q_full) = pf.calc_power_vecs(system, vmag_full, delta_full, g, b)

#     jacobian = pf.calc_jacobian(system, vmag_full, delta_full, g, b, p_full, q_full)

#     jacobian_calc = pf.jacobian_calc_simplify(system, jacobian)

#     (del_p, del_q) = pf.calc_mismatch_vecs(system, p, q)
    
#     y = np.row_stack((del_p, del_q))


#     # print("\nIteration %d:\n" % i)
#     # print("delta:\n",delta * 180/np.pi)
#     # print("vmag:\n",vmag)
#     # print("mismatch vector:\n", y)
#     # print("Jacobian:\n", jacobian_calc)

#     if pf.check_convergence(y, tolerance):
#         print("Power flow converged at %d iterations (tolerance of %f).\n" % (i, tolerance))
#         print("Mismatch vector (P injections)\n", del_p)
#         print("Mismatch vector (Q injections)\n", del_q)
#         print("\nTable of results (power values are injections):\n")
#         typelist = ['' for i in range(n_buses)]
#         typelist[system.get('slack').get('bus')] = 'SLACK'
    
#         for gen in system.get('generators'):
#             k = gen.get('bus')
#             typelist[k] = gen.get('type').upper()
        
#         for i in range(n_buses):
#             if typelist[i] == '':
#                 typelist[i] = 'PQ'
        
#         d = {'vmag_pu':vmag_full.flatten(), 'delta_deg':delta_full.flatten()*180/np.pi, 'p_pu':p_full.flatten(), 'q_pu':q_full.flatten(), 'type':typelist}
#         df = pd.DataFrame(data=d, index = np.arange(n_buses))
#         df.index.name = 'bus'
#         print(df)
#         break
    
#     elif i == iteration_limit:
#         print("Power flow did not converge after %d iterations (tolerance of %f).\n" % (i, tolerance))

