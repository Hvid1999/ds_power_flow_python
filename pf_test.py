import power_flow_newton_raphson as pf
import pandapower as pp
import pandapower.networks as nw
import numpy as np
import pandas as pd
pd.options.display.float_format = '{:.6f}'.format #avoid scientific notation
###################################################################################################

#Cases with slack bus on other index than 1 (0)
#case5
#case24_ieee_rts
#case39
#...

network = nw.case4gs()
#network = nw.case5()  #this case may be bugged...
#network = nw.case6ww()
#network = nw.case9()
#network = nw.case14()
#network = nw.case24_ieee_rts() #weird results compared to pandapower and no convergence for qlims
#network = nw.case30()
#network = nw.case_ieee30()
#network = nw.case33bw()
#network = nw.case39()
#network = nw.case57()


#Cases source: 
#https://pandapower.readthedocs.io/en/v2.8.0/networks/power_system_test_cases.html


#Function loading system information from PandaPower network
#system = pf.load_pandapower_case(network)

e_q_lims = True #enforce q-limits True/False

baseMVA = network.sn_mva #base power for system
freq = network.f_hz

pp.runpp(network, enforce_q_lims=e_q_lims) #run power flow
ybus = network._ppc["internal"]["Ybus"].todense() #extract Ybus after running power flow
gen = network.gen
load = network.load
slack = network.ext_grid
buses = network.bus #bus parameters
lines = network.line #line parameters
shunts = network.shunt #information about shunts

#NOTE! if shunts are not included in the loads in PandaPower, 
#they need to be included explicitly in the code.

#Saving PandaPower results and per-unitizing power values
pandapower_results = network.res_bus
pandapower_results['p_pu'] = pandapower_results.p_mw/baseMVA
pandapower_results['q_pu'] = pandapower_results.q_mvar/baseMVA
pandapower_results = pandapower_results[['vm_pu','va_degree','p_pu','q_pu']]


#loading line and transformer parameters
pf_line_flows = network.res_line #line flows from pandapower results


#loading slack bus information
slack_dict = {'bus':slack.bus[0], 'vset':slack.vm_pu[0], 'pmin':slack.min_p_mw[0]/baseMVA,
              'pmax':slack.max_p_mw[0]/baseMVA, 'qmin':slack.min_q_mvar[0]/baseMVA, 
              'qmax':slack.max_q_mvar[0]/baseMVA}


#Setup system dictionary
system = {'admmat':ybus,'slack':slack_dict, 'buses':[], 'generators':[],'loads':[],
          'lines':[],'iteration_limit':15,'tolerance':1e-3, 's_base':baseMVA, 'frequency':freq}

#initializing empty lists
gen_list = []
load_list = []
bus_list = []
line_list = []

#Fill lists of generator and load dictionaries based on the loaded generator and load information from PandaPower
#Per-unitizing the values according to the power base
for i in range(len(gen.index)):
    gen_list.append({'type':'pv', 'bus':gen.bus[i], 'vset':gen.vm_pu[i], 'pset':gen.p_mw[i]/baseMVA,
                     'qset':None, 'qmin':gen.min_q_mvar[i]/baseMVA, 'qmax':gen.max_q_mvar[i]/baseMVA,
                     'pmin':gen.min_p_mw[i]/baseMVA, 'pmax':gen.max_p_mw[i]/baseMVA})

for i in range(len(load.index)):
    load_list.append({'bus':load.bus[i], 'p':load.p_mw[i]/baseMVA, 'q':load.q_mvar[i]/baseMVA})
    
for i in range(len(buses.index)):
    bus_list.append({'v_max':buses.max_vm_pu[i], 'v_min':buses.min_vm_pu[i], 'v_base':buses.vn_kv[i],
                     'zone':buses.zone[i], 'index':buses.name[i]})
    
for i in range(len(lines.index)):
    line_list.append({'from':lines.from_bus[i], 'to':lines.to_bus[i], 'length':lines.length_km[i], 
                      'ampacity':lines.max_i_ka[i], 'g_mu_s_per_km':lines.g_us_per_km[i], 'c_nf_per_km':lines.c_nf_per_km[i],
                      'r_per_km':lines.r_ohm_per_km[i], 'x_per_km':lines.x_ohm_per_km[i], 
                      'parallel':lines.parallel[i]})

system.update({'generators':gen_list})
system.update({'loads':load_list})
system.update({'buses':bus_list})
system.update({'lines':line_list})


#%%
pf_results = pf.run_newton_raphson(system, enforce_q_limits=e_q_lims)
print('\nPandaPower results:\n')
print(pandapower_results)

#%%
vtest = pf_results.get('bus_results')['vmag_pu']
vtest = pd.Series.to_numpy(vtest)

deltatest = pf_results.get('bus_results')['delta_deg']
deltatest = pd.Series.to_numpy(deltatest) * np.pi / 180



#test code for current calculations
line = system.get('lines')[0]
l = line.get('length')

from_idx = line.get('from')
to_idx = line.get('to')
y_shunt = complex(0,2*np.pi*freq*line.get('c_nf_per_km')*1e-9) * l
z_line = complex(line.get('r_per_km'), line.get('x_per_km')) * l

(s, p, q) = pf.calc_power_from_to(system, vtest, deltatest, from_idx, to_idx)
print('P: %f MW \nQ: %f MVAR'  % (p * baseMVA, q * baseMVA))

# I_12 = (V_1 - V_2) / (Z_12) + V_1 / Y_sh / 2

v_1 = complex(vtest[from_idx]*np.cos(deltatest[from_idx]), vtest[from_idx]*np.sin(deltatest[from_idx]))
v_2 = complex(vtest[to_idx]*np.cos(deltatest[to_idx]), vtest[to_idx]*np.sin(deltatest[to_idx]))

i_from_to = np.abs((v_1 - v_2) / z_line + v_1 / (y_shunt / 2))
# Way too high value... hmmm

#%%
print(pf_line_flows)

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

