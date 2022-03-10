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
#network = nw.case_ieee30()
#network = nw.case33bw()
network = nw.case39()  #New England 39-bus system??
#network = nw.case57()

#Cases source: 
#https://pandapower.readthedocs.io/en/v2.8.0/networks/power_system_test_cases.html

e_q_lim = True

#Contingency testing
#network.load['in_service'][0] = False
#network.line['in_service'][2] = False

#function loading test case system information and power flow results from PandaPower
(system, pandapower_results) = pf.load_pandapower_case(network, enforce_q_limits = e_q_lim)


#%%
results = pf.run_newton_raphson(system, enforce_q_limits = e_q_lim)
print('\nPandaPower results:\n')
print(pandapower_results)
