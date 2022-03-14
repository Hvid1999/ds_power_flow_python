import power_flow_newton_raphson as pf
import pandapower as pp
import pandapower.networks as nw
import numpy as np
import pandas as pd
pd.options.display.float_format = '{:.6f}'.format #avoid scientific notation

#==============================================================================
#network = nw.case4gs()
network = nw.case5()
#network = nw.case6ww()
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

e_q_lim = True
dist_slack = True
slack_gens = np.array([]) #generator list indices (generators 0, 1 .. G)
participation_factors = np.array([]) 

#Contingency testing
#network.load['in_service'][0] = False
#network.line['in_service'][2] = False

#function loading test case system information and power flow results from PandaPower
(system, pandapower_results) = pf.load_pandapower_case(network, enforce_q_limits = e_q_lim,
                                                       distributed_slack = dist_slack, slack_gens = slack_gens,
                                                       participation_factors = participation_factors)



results = pf.run_power_flow(system, enforce_q_limits=e_q_lim, distributed_slack=dist_slack)


#Plan for adjusting code to incorporate distributed slack:
    
#==============================================================================
#To Do (listed by priority):
    
#Add a function to check real power generator limit violations for distributed slack
#consider what should be done in this case

#Investigate consistency of results


#==============================================================================

#DONE:
#Add function which implements the system loss function

#Add function to load participation factors

#Adjust the Jacobian function for distributed slack

#Alter the system loading function
    #Generator list includes the external grid slack bus object
    #Generators have a 'slack' boolean parameter
    #Generators have a participation factor attributed
    #The power setpoint of the original slack bus generator is (load - generation)
    #Original slack bus treated as PV-bus

#Tweak functions to work for both distributed and single slack power flow

#Adjust convergence criterium of power flow 

#At the end of Newton-Raphson for distributed slack, print the mismatch vectors,
#the system slack and how much of the slack is distributed to each bus
#to see if it aligns with mismatches

#==============================================================================







