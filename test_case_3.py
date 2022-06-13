import ds_power_flow as pf
import pandapower.networks as nw
import numpy as np
import pandas as pd

pd.options.display.float_format = '{:.6f}'.format #avoid scientific notation when printing dataframes

# network = nw.case39()
network = pf.new_england_39_new_voltages(nw.case39())
network.gen['vm_pu'][5] = 1.058

#Contingency
pf.panda_disable_bus(network, 31)
# pf.panda_disable_bus(network, 38)


# Scaling line resistance to obtain more realistic system losses
network.line['r_ohm_per_km'] = network.line['r_ohm_per_km'] * 5.0 #around 3%
# network.line['r_ohm_per_km'] = network.line['r_ohm_per_km'] * 7.0

# desc = "Low Losses (Standard Line Resistances)"
desc = "Medium Losses - Upscaled Line Resistance (Factor 5.0)"
# desc = "High Losses - Upscaled Line Resistance (Factor 7.0)"

# slack_gens = np.arange(0,10)
slack_gens = np.array([])
# slack_gens = np.array([0,1,2,3,4,5,6,7,8])
#Using the standard factors established in test one
participation_factors = np.array([0.21895328, 0.06576625, 0.07380888, 0.13726686, 0.10695025, 0.06994027, 0.12210856, 0.05741822, 0.08806163, 0.0597258])
participation_factors = participation_factors / np.sum(participation_factors) #renormalize


#function loading test case system information and power flow results from PandaPower
system_ss = pf.load_pandapower_case(network, enforce_q_limits = True, distributed_slack = False, 
                                    slack_gens = slack_gens, participation_factors = participation_factors)[0]
system_ds = pf.load_pandapower_case(network, enforce_q_limits = True, distributed_slack = True, 
                                    slack_gens = slack_gens, participation_factors = participation_factors)[0]


pf.new_england_case_line_fix(system_ss)
pf.new_england_case_line_fix(system_ds)

gens = system_ds.get('generators').copy()

p_fact_tweak = np.copy(participation_factors)


results_ss = pf.run_power_flow(system_ss, enforce_q_limits=True)
results_ds = pf.run_power_flow(system_ds, enforce_q_limits=True)
#%%
# name = ('Case 3\n%s\nSingle Slack Bus') % desc
pf.plot_results(system_ss, results_ss, angle=True, axis_values=[-0.87, 1.08, 40, 120, 120])
# name = ('Case 4\n%s\nDistributed Slack Bus') % desc
pf.plot_results(system_ds, results_ds, angle=True, axis_values=[-0.87, 1.08, 40, 120, 120])


# pf.plot_result_comparison(results_ss, results_ds, angle=True, name = ('Case 4\n%s\nSingle Slack vs. Distributed Slack') % desc)

