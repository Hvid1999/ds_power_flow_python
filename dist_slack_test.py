import power_flow_newton_raphson as pf
import pandapower as pp
import pandapower.networks as nw
import numpy as np
import pandas as pd
pd.options.display.float_format = '{:.6f}'.format #avoid scientific notation

#==============================================================================
#network = nw.case4gs()
#network = nw.case5()
network = nw.case6ww()
#network = nw.case9()
#network = nw.case14()
#network = nw.case24_ieee_rts()
#network = nw.case30()
#network = nw.case_ieee30()
#network = nw.case33bw()
#network = nw.case39()  #New England 39-bus system
#network = nw.case57()

#Cases source: 
#https://pandapower.readthedocs.io/en/v2.8.0/networks/power_system_test_cases.html
#==============================================================================

e_q_lim = False

#Contingency testing
#network.load['in_service'][0] = False
#network.line['in_service'][2] = False

#function loading test case system information and power flow results from PandaPower
#(system, pandapower_results) = pf.load_pandapower_case(network, enforce_q_limits = e_q_lim)


#%%
#Plan for adjusting code to incorporate distributed slack:
    #Alter the system loading function
        #Generator list includes the external grid slack bus object
        #Generators have a 'slack' boolean parameter
        #Generators have a participation factor attributed
        #The power setpoint of the original slack bus generator is (load - generation)
        #Original slack bus treated as PV-bus
        
    #Adjust the Jacobian function for distributed slack
    
    #Adjust convergence criterium of power flow to not be based on nominal
    #setpoints directly (mismatch vector), but on the change from the last value
    
    #Add function which implements the system loss function
    
    #Add function to load participation factors
    
    #Perhaps implement a boolean input 'dist_slack' that when false
    #sets the participation factors such that the single slack bus problem
    #is recovered. This makes the function library primarily focused on
    #distributed slack, but also featuring a simple single slack bus solver
    
#%%
#Experimental code adjustments

baseMVA = network.sn_mva #base power for system
freq = network.f_hz

pp.runpp(network, enforce_q_lims=e_q_lim) #run power flow
ybus = network._ppc["internal"]["Ybus"].todense() #extract Ybus after running power flow
gen = network.gen #voltage controlled generators
sgen = network.sgen #static generators (PQ)
load = network.load
slack = network.ext_grid
buses = network.bus #bus parameters
lines = network.line #line parameters
shunts = network.shunt #information about shunts


#Saving PandaPower results and per-unitizing power values
pandapower_results = network.res_bus
pandapower_results['p_pu'] = pandapower_results.p_mw/baseMVA
pandapower_results['q_pu'] = pandapower_results.q_mvar/baseMVA
pandapower_results = pandapower_results[['vm_pu','va_degree','p_pu','q_pu']]


#Setup system dictionary
system = {'admmat':ybus, 'buses':[],'iteration_limit':15,'tolerance':1e-3, 
          's_base':baseMVA, 'frequency':freq}

#initializing empty lists
gen_list = []
load_list = []
bus_list = []
line_list = []
shunt_list = []

#Fill lists of generator and load dictionaries based on the loaded generator and load information from PandaPower
#Per-unitizing the values according to the power base


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
    
#The setpoint original slack bus generator is difference between total load and total generation
load_sum = 0
gen_sum = 0

for load in load_list:
    load_sum += load.get('p')

for gen in gen_list:
    gen_sum += gen.get('pset')

slack_pset = load_sum - gen_sum

#Adding original slack bus generator as PV-bus generator
slack_dict = {'type':'pv','bus':slack.bus[0], 'vset':slack.vm_pu[0], 'pset':slack_pset,
              'qset':None, 'qmin':slack.min_q_mvar[0]/baseMVA, 'qmax':slack.max_q_mvar[0]/baseMVA,
              'pmin':slack.min_p_mw[0]/baseMVA, 'pmax':slack.max_p_mw[0]/baseMVA, 
              'in_service':slack.in_service[0]}

gen_list.append(slack_dict) 

#sort list of generator dictionaries by bus placement after loading both PV and PQ generators
gen_list = sorted(gen_list, key=lambda d: d['bus'])



system.update({'generators':gen_list})
system.update({'loads':load_list})
system.update({'shunts':shunt_list})
system.update({'buses':bus_list})
system.update({'lines':line_list})




