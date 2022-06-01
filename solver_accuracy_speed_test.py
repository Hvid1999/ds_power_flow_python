import ds_power_flow as pf
import pandapower.networks as nw
import numpy as np
import pandas as pd
import pandapower as pp
import time

pd.options.display.float_format = '{:.6f}'.format #avoid scientific notation when printing dataframes

network = pf.new_england_39_new_voltages(nw.case39())
enforce_q_limits = True
slack_gens = np.array([1])
participation_factors = np.array([0,1,0,0,0,0,0,0,0,0])

n_comp = 300
#%%
#Testing accuracy

(system, pandapower_results) = pf.load_pandapower_case(network, enforce_q_limits = enforce_q_limits,
                                                       distributed_slack = False, slack_gens = slack_gens,
                                                       participation_factors = participation_factors)
system.update({'tolerance':1e-08})
results = pf.run_power_flow(system, enforce_q_limits=enforce_q_limits, print_results=False)


system_ds = pf.load_pandapower_case(network, enforce_q_limits = enforce_q_limits,
                                                       distributed_slack = True, slack_gens = slack_gens,
                                                       participation_factors = participation_factors)[0]
system_ds.update({'tolerance':1e-08})
results_ds = pf.run_power_flow(system_ds, enforce_q_limits=enforce_q_limits, print_results=False)

bus_res_ss = results.get('bus_results')[['vmag_pu','delta_deg','p_pu','q_pu']]
bus_res_ds = results_ds.get('bus_results')[['vmag_pu','delta_deg','p_pu','q_pu']]

#To easily compare to pandapower, which uses load convention and not injections
bus_res_ss['p_pu'] *= -1
bus_res_ss['q_pu'] *= -1
bus_res_ds['p_pu'] *= -1
bus_res_ds['q_pu'] *= -1
bus_res_ss = bus_res_ss.rename(columns={'vmag_pu':'vm_pu', 'delta_deg':'va_degree'})
bus_res_ds = bus_res_ds.rename(columns={'vmag_pu':'vm_pu', 'delta_deg':'va_degree'})

bus_diff_ss = pandapower_results - bus_res_ss 
bus_diff_ds = pandapower_results - bus_res_ds 

average_diff_ss = bus_diff_ss.mean(axis=0)
average_diff_ds = bus_diff_ds.mean(axis=0)


#%%
#Testing speed

network = pf.new_england_39_new_voltages(nw.case39())
enforce_q_limits = True

#PandaPower
pp_time_sum = 0
for i in range(n_comp):
    t1 = time.time()
    pp.runpp(network, enforce_q_lims = enforce_q_limits, trafo_model='pi', trafo_loading='power', max_iteration=25)
    t2 = time.time()
    pp_time_sum += t2-t1
pp_time = pp_time_sum / n_comp

#Single slack with Q-lims and low tolerance
pf_time_sum = 0
for i in range(n_comp):
    system = pf.load_pandapower_case(network, enforce_q_limits = enforce_q_limits,
                                                           distributed_slack = False, slack_gens = slack_gens,
                                                           participation_factors = participation_factors)[0]
    system.update({'tolerance':1e-08})
    t1 = time.time()
    results = pf.run_power_flow(system, enforce_q_limits=enforce_q_limits, print_results=False, print_bus_type=False)
    t2 = time.time()
    pf_time_sum += t2-t1
pf_time = pf_time_sum / n_comp

#Single slack with Q-lims and increased tolerance
pf_time_sum_2 = 0
for i in range(n_comp):
    system = pf.load_pandapower_case(network, enforce_q_limits = enforce_q_limits,
                                                           distributed_slack = False, slack_gens = slack_gens,
                                                           participation_factors = participation_factors)[0]
    system.update({'tolerance':1e-03})
    t1 = time.time()
    results = pf.run_power_flow(system, enforce_q_limits=enforce_q_limits, print_results=False, print_bus_type=False)
    t2 = time.time()
    pf_time_sum_2 += t2-t1
pf_time_2 = pf_time_sum_2 / n_comp

#Distributed slack with Q-lims and increased tolerance
pf_time_sum_3 = 0
for i in range(n_comp):
    system_ds = pf.load_pandapower_case(network, enforce_q_limits = enforce_q_limits,
                                                           distributed_slack = True, slack_gens = np.array([1]),
                                                           participation_factors = participation_factors)[0]
    system_ds.update({'tolerance':1e-03})
    t1 = time.time()
    results = pf.run_power_flow(system_ds, enforce_q_limits=enforce_q_limits, print_results=False, print_bus_type=False)
    t2 = time.time()
    pf_time_sum_3 += t2-t1
pf_time_3 = pf_time_sum_3 / n_comp

print('Speed results (Q-limits enforced):')
print('pandapower: %f s' %pp_time)
print('Custom solver (tolerance of 1e-08): %f s' %pf_time)
print('Custom solver (tolerance of 1e-03): %f s' %pf_time_2)
print('Custom solver (DS - tolerance of 1e-03): %f s' %pf_time_3)

speed_results_q_lim = {'pandapower':pp_time, 'Custom 1e-08':pf_time, 'Custom 1e-03':pf_time_2, 'Custom (DS) 1e-03':pf_time_3}
#%%
#Speed: Not enforcing q-limits
network = pf.new_england_39_new_voltages(nw.case39())
enforce_q_limits = False
system = pf.load_pandapower_case(network, enforce_q_limits = enforce_q_limits,
                                                       distributed_slack = False, slack_gens = slack_gens,
                                                       participation_factors = participation_factors)[0]
system.update({'tolerance':1e-08})

pp_time_sum = 0
for i in range(n_comp):
    t1 = time.time()
    pp.runpp(network, enforce_q_lims = enforce_q_limits, trafo_model='pi', trafo_loading='power', max_iteration=25)
    t2 = time.time()
    pp_time_sum += t2-t1
pp_time = pp_time_sum / n_comp

pf_time_sum = 0
for i in range(n_comp):
    system = pf.load_pandapower_case(network, enforce_q_limits = enforce_q_limits,
                                                           distributed_slack = False, slack_gens = slack_gens,
                                                           participation_factors = participation_factors)[0]
    system.update({'tolerance':1e-08})
    t1 = time.time()
    results = pf.run_power_flow(system, enforce_q_limits=enforce_q_limits, print_results=False, print_bus_type=False)
    t2 = time.time()
    pf_time_sum += t2-t1
pf_time = pf_time_sum / n_comp


pf_time_sum_2 = 0
for i in range(n_comp):
    system = pf.load_pandapower_case(network, enforce_q_limits = enforce_q_limits,
                                                           distributed_slack = False, slack_gens = slack_gens,
                                                           participation_factors = participation_factors)[0]
    system.update({'tolerance':1e-03})
    t1 = time.time()
    results = pf.run_power_flow(system, enforce_q_limits=enforce_q_limits, print_results=False, print_bus_type=False)
    t2 = time.time()
    pf_time_sum_2 += t2-t1
pf_time_2 = pf_time_sum_2 / n_comp


pf_time_sum_3 = 0
for i in range(n_comp):
    system_ds = pf.load_pandapower_case(network, enforce_q_limits = enforce_q_limits,
                                                           distributed_slack = True, slack_gens = np.array([1]),
                                                           participation_factors = participation_factors)[0]
    system_ds.update({'tolerance':1e-03})
    t1 = time.time()
    results = pf.run_power_flow(system_ds, enforce_q_limits=enforce_q_limits, print_results=False, print_bus_type=False)
    t2 = time.time()
    pf_time_sum_3 += t2-t1
pf_time_3 = pf_time_sum_3 / n_comp

print('Speed results (Q-limits NOT enforced):')
print('pandapower: %f s' %pp_time)
print('Custom solver (tolerance of 1e-08): %f s' %pf_time)
print('Custom solver (tolerance of 1e-03): %f s' %pf_time_2)
speed_results_non_q_lim = {'pandapower':pp_time, 'Custom 1e-08':pf_time, 'Custom 1e-03':pf_time_2, 'Custom (DS) 1e-03':pf_time_3}
