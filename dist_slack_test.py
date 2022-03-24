import power_flow_newton_raphson as pf
import pandapower.networks as nw
import numpy as np
import pandas as pd

pd.options.display.float_format = '{:.6f}'.format #avoid scientific notation when printing dataframes

#==============================================================================
#network = nw.case4gs()
#network = nw.case5()
#network = nw.case6ww()
#network = nw.case9()
#network = nw.case14()
#network = nw.case24_ieee_rts()
#network = nw.case30()
#network = nw.case_ieee30()
#network = nw.case33bw()
network = nw.case39()  #New England 39-bus system

#Cases source: 
#https://pandapower.readthedocs.io/en/v2.8.0/networks/power_system_test_cases.html
#==============================================================================

enforce_q_limits = True
distributed_slack = True
slack_gens = np.array([0,1,2,3,5,6,7,8,9]) #generator list indices (generators 0, 1 .. G)
participation_factors = np.array([0.7, 0.01, 0.07, 0.02, 0.03, 0.02, 0.02, 0.03, 0.1]) 

#Contingency testing
#network.load['in_service'][0] = False
# network.line['in_service'][33] = False
# network.line['in_service'][34] = False
# network.gen['in_service'][9] = False


#function loading test case system information and power flow results from PandaPower
(system, pandapower_results) = pf.load_pandapower_case(network, enforce_q_limits = enforce_q_limits,
                                                       distributed_slack = distributed_slack, slack_gens = slack_gens,
                                                       participation_factors = participation_factors)

#%%

#Vary load at the stated indices in the loads list of dictionaries
#and distribute the mismatch across the generators according to participation factors
pf.load_variation(system, [10,20], [1.10, 1.25])

results = pf.run_power_flow(system, enforce_q_limits=enforce_q_limits, distributed_slack=distributed_slack)

pf.plot_results(system, results)

#%%
    
#==============================================================================
#To Do (listed by priority):



#Consider the case of one generator being entered as slack generator while the distributed slack bool 
#is true... Perhaps that case should change the original slack bus and run single slack power flow?  


#==============================================================================

#%%

