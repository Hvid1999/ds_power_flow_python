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
dist_slack = True
slack_gens = np.array([]) #generator list indices (generators 0, 1 .. G)

#Contingency testing
#network.load['in_service'][0] = False
#network.line['in_service'][2] = False

#function loading test case system information and power flow results from PandaPower
(system, pandapower_results) = pf.load_pandapower_case(network, enforce_q_limits = e_q_lim,
                                                       distributed_slack = dist_slack)


#Plan for adjusting code to incorporate distributed slack:
    
#==============================================================================
#To Do:

#Tweak functions to work for both distributed and single slack power flow
    
#Adjust the Jacobian function for distributed slack

#Adjust convergence criterium of power flow to not be based on nominal
#setpoints directly (mismatch vector), but on the change from the last value

#==============================================================================

#DONE:
#Add function which implements the system loss function

#Add function to load participation factors

#Alter the system loading function
    #Generator list includes the external grid slack bus object
    #Generators have a 'slack' boolean parameter
    #Generators have a participation factor attributed
    #The power setpoint of the original slack bus generator is (load - generation)
    #Original slack bus treated as PV-bus

#==============================================================================


#%%

#Augmented to include distributed slack:

(n_buses, g, b) = pf.process_admittance_mat(system)

(vmag, delta, vmag_full, delta_full) = pf.init_voltage_vecs(system)

#%%

# Functions not yet changed:

# (p, q, p_full, q_full) = calc_power_vecs(system, vmag_full, delta_full, g, b)

# jacobian = calc_jacobian(system, vmag_full, delta_full, g, b, p_full, q_full)

# jacobian_calc = jacobian_calc_simplify(system, jacobian)

# pf.check_convergence(y, threshold)

# and more...
