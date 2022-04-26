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
#network = nw.case39() #New England 39-bus system

network = pf.new_england_39_new_voltages(nw.case39())


#Cases source: 
#https://pandapower.readthedocs.io/en/v2.8.0/networks/power_system_test_cases.html
#==============================================================================

# Scaling line resistance to obtain more realistic system losses
network.line['r_ohm_per_km'] = network.line['r_ohm_per_km'] * 3.5 #around 2%
# network.line['r_ohm_per_km'] = network.line['r_ohm_per_km'] * 7.0

# desc = "Low Losses"
desc = "Medium Losses - Upscaled Line Resistance (Factor 3.5)"
# desc = "High Losses - Upscaled Line Resistance (Factor 7.0)"

enforce_q_limits = True
distributed_slack = True
#slack_gens = np.array([0,1,2,3,5,6,7,8,9]) #generator list indices (generators 0, 1 .. G)
#participation_factors = np.array([0.7, 0.01, 0.07, 0.02, 0.03, 0.02, 0.02, 0.03, 0.1]) 
slack_gens = np.array([])
participation_factors = np.array([])


#Contingency testing
#network.load['in_service'][0] = False
# network.line['in_service'][33] = False
network.line['in_service'][0] = False
# network.gen['in_service'][9] = False


#function loading test case system information and power flow results from PandaPower
(system, pandapower_results) = pf.load_pandapower_case(network, enforce_q_limits = enforce_q_limits,
                                                       distributed_slack = distributed_slack, slack_gens = slack_gens,
                                                       participation_factors = participation_factors)


#Important: If the New England 39 Bus system is loaded, uncomment the code below to fix 
#           the transformer (bus 22 to 35) loaded as a line.
#Also, all busses are set as 345 kV, which is slightly inaccurate, but it shouldn't make much of a 
#difference. Maybe worth testing.

pf.new_england_case_line_fix(system)



#Vary load at the stated indices in the loads list of dictionaries
#and distribute the mismatch across the generators according to participation factors
#pf.load_variation(system, [10,20], [1.10, 1.25])

results = pf.run_power_flow(system, enforce_q_limits=enforce_q_limits, distributed_slack=distributed_slack)

pf.plot_results(system, results)


#%%
#Evaluating line contingencies to find troublesome cases
network = pf.new_england_39_new_voltages(nw.case39())

invalid_lines_cases = []
line = network.line

for i in range(len(line.index)):
    try:
        network = pf.new_england_39_new_voltages(nw.case39())
        network.line['r_ohm_per_km'] = network.line['r_ohm_per_km'] * 3.5 #around 2%
        network.line['in_service'][i] = False
        system = pf.load_pandapower_case(network, enforce_q_limits = True,
                                                               distributed_slack = True, slack_gens = np.arange(0,10),
                                                               participation_factors = np.array([]))[0]
        pf.new_england_case_line_fix(system)
        results = pf.run_power_flow(system, enforce_q_limits=True, distributed_slack=True, print_results=False)
        frombus = line.from_bus[i]
        tobus = line.to_bus[i]
        
        pf.plot_results(system, results, name = ('Line %d\nBus %d to %d\nEqual participation' % (i, frombus, tobus)))
    except:
        invalid_lines_cases.append(i)
            
#%%
#Evaluating load contingencies to find troublesome cases
network = pf.new_england_39_new_voltages(nw.case39())
ds = False
if ds:
    desc = 'Equal participation'
else:
    desc = 'Single slack'
load = network.load
for i in range(len(network.load.index)):
    try:
        network = pf.new_england_39_new_voltages(nw.case39())
        network.line['r_ohm_per_km'] = network.line['r_ohm_per_km'] * 3.5 #around 2%
        system = pf.load_pandapower_case(network, enforce_q_limits = True,
                                                               distributed_slack = ds, 
                                                               slack_gens = np.array([0,1,3,4,5,6,7,8,9]),
                                                               participation_factors = np.array([]))[0]
        pf.new_england_case_line_fix(system)
        pf.load_variation(system, np.array([i]), scalings=np.ones(1)*0.0, const_pf=True)
        results = pf.run_power_flow(system, enforce_q_limits=True, distributed_slack=ds, print_results=False)
        bus = load.bus[i]
        pf.plot_results(system, results, name = ('Load %d\nBus %d\n%s' % (i, bus, desc)))
    except:
        1+1

#%%
#Evaluating generator contingencies to find troublesome cases
network = pf.new_england_39_new_voltages(nw.case39())
ds = True
if ds:
    desc = 'Equal participation'
else:
    desc = 'Single slack'
gen = network.gen
for i in range(9):
    # try:
    if i > 0:
        slack_gens = np.delete(np.arange(0,10), i+1)
    else: 
        slack_gens = np.delete(np.arange(0,10), i)
    bus = gen.bus[i]
    network = pf.new_england_39_new_voltages(nw.case39())
    network.line['r_ohm_per_km'] = network.line['r_ohm_per_km'] * 3.5 #around 2%
    pf.panda_disable_bus(network, bus)
    system = pf.load_pandapower_case(network, enforce_q_limits = True,
                                                           distributed_slack = ds, 
                                                           slack_gens = slack_gens,
                                                           participation_factors = np.array([]))[0]
    pf.new_england_case_line_fix(system)
    results = pf.run_power_flow(system, enforce_q_limits=True, distributed_slack=ds, print_results=False)
    pf.plot_results(system, results, name = ('Bus %d\n%s' % (bus, desc)))
    # except:
        # 1+1            

#%%
#Evaluating bus contingencies to find troublesome cases
network = pf.new_england_39_new_voltages(nw.case39())
ds = True
invalid_buses = []
if ds:
    desc = 'Equal participation'
else:
    desc = 'Single slack'
gen = network.gen
for i in range(0,29):
    try:
        network = pf.new_england_39_new_voltages(nw.case39())
        network.line['r_ohm_per_km'] = network.line['r_ohm_per_km'] * 3.5 #around 2%
        pf.panda_disable_bus(network, i)
        system = pf.load_pandapower_case(network, enforce_q_limits = True,
                                                               distributed_slack = ds, 
                                                               slack_gens = np.array([0,1,3,4,5,6,7,8,9]),
                                                               participation_factors = np.array([]))[0]
        pf.new_england_case_line_fix(system)
        results = pf.run_power_flow(system, enforce_q_limits=True, distributed_slack=ds, print_results=False)
        pf.plot_results(system, results, name = ('Bus %d\n%s' % (i, desc)))
    except:
        invalid_buses.append(i)          

