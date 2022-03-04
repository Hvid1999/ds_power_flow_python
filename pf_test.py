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

#network = nw.case4gs()
#network = nw.case5()
#network = nw.case6ww()
#network = nw.case9()
#network = nw.case14()
#network = nw.case24_ieee_rts()
#network = nw.case30()
network = nw.case_ieee30()
#network = nw.case33bw()
#network = nw.case39()
#network = nw.case57()

#Cases source: 
#https://pandapower.readthedocs.io/en/v2.8.0/networks/power_system_test_cases.html


#Function loading system information from PandaPower network
#Will contain most of the code below
#system = pf.load_pandapower_case(network, e_q_lims = True)

e_q_lims = True #enforce q-limits True/False

baseMVA = network.sn_mva #base power for system
freq = network.f_hz

pp.runpp(network, enforce_q_lims=e_q_lims) #run power flow
ybus = network._ppc["internal"]["Ybus"].todense() #extract Ybus after running power flow
gen = network.gen #voltage controlled generators
sgen = network.sgen #static generators (PQ)
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
system = {'admmat':ybus,'slack':slack_dict, 'buses':[], 'generators':[],'loads':[], 'shunts':[],
          'lines':[],'iteration_limit':15,'tolerance':1e-3, 's_base':baseMVA, 'frequency':freq}

#initializing empty lists
gen_list = []
load_list = []
bus_list = []
line_list = []
shunt_list = []

#Fill lists of generator and load dictionaries based on the loaded generator and load information from PandaPower
#Per-unitizing the values according to the power base

#Voltage controlled generators
for i in range(len(gen.index)):
    gen_list.append({'type':'pv', 'bus':gen.bus[i], 'vset':gen.vm_pu[i], 'pset':gen.p_mw[i]/baseMVA,
                     'qset':None, 'qmin':gen.min_q_mvar[i]/baseMVA, 'qmax':gen.max_q_mvar[i]/baseMVA,
                     'pmin':gen.min_p_mw[i]/baseMVA, 'pmax':gen.max_p_mw[i]/baseMVA, 
                     'in_service':gen.in_service[i]})

#Static generators
for i in range(len(sgen.index)):
    gen_list.append({'type':'pq', 'bus':sgen.bus[i], 'vset':None, 'pset':sgen.p_mw[i]/baseMVA,
                     'qset':sgen.q_mvar[i]/baseMVA, 'qmin':sgen.min_q_mvar[i]/baseMVA,
                     'qmax':sgen.max_q_mvar[i]/baseMVA, 'pmin':sgen.min_p_mw[i]/baseMVA, 
                     'pmax':sgen.max_p_mw[i]/baseMVA, 'in_service':sgen.in_service[i]})

#sort list of generator dictionaries by bus placement after loading both PV and PQ generators
gen_list = sorted(gen_list, key=lambda d: d['bus'])


for i in range(len(load.index)):
    load_list.append({'bus':load.bus[i], 'p':load.p_mw[i]/baseMVA, 'q':load.q_mvar[i]/baseMVA, 
                      'in_service':load.in_service[i]})
    
for i in range(len(buses.index)):
    bus_list.append({'v_max':buses.max_vm_pu[i], 'v_min':buses.min_vm_pu[i], 'v_base':buses.vn_kv[i],
                     'zone':buses.zone[i], 'index':buses.name[i]})
    
for i in range(len(lines.index)):
    line_list.append({'from':lines.from_bus[i], 'to':lines.to_bus[i], 'length':lines.length_km[i], 
                      'ampacity':lines.max_i_ka[i], 'g_mu_s_per_km':lines.g_us_per_km[i], 'c_nf_per_km':lines.c_nf_per_km[i],
                      'r_ohm_per_km':lines.r_ohm_per_km[i], 'x_ohm_per_km':lines.x_ohm_per_km[i], 
                      'parallel':lines.parallel[i], 'in_service':lines.in_service[i]})
    
for i in range(len(shunts.index)):
    #Note: The PandaPower shunt power values are CONSUMPTION (load convention)
    shunt_list.append({'bus':shunts.bus[i], 'q_pu':shunts.q_mvar[i]/baseMVA,
                       'p_pu':shunts.p_mw[i]/baseMVA, 'v_rated':shunts.vn_kv[i], 
                       'in_service':shunts.in_service[i]})

#Shunts are basically handled as loads in this code.
#PandaPower implements shunts in a different way - they essentially do not 
#affect bus voltages, but instead their power consumption is calculated 
#and added to the bus power consumption
#more on this: https://pandapower.readthedocs.io/en/v2.8.0/elements/shunt.html 

system.update({'generators':gen_list})
system.update({'loads':load_list})
system.update({'shunts':shunt_list})
system.update({'buses':bus_list})
system.update({'lines':line_list})


#%%
results = pf.run_newton_raphson(system, enforce_q_limits=e_q_lims)
print('\nPandaPower results:\n')
print(pandapower_results)


