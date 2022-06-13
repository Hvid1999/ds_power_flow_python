import ds_power_flow as pf
import pandapower.networks as nw
import numpy as np
import pandas as pd

pd.options.display.float_format = '{:.6f}'.format #avoid scientific notation when printing dataframes

#system tweaks
network = pf.new_england_39_new_voltages(nw.case39())
network.gen['vm_pu'][5] = 1.058

# Scaling line resistance to obtain more realistic system losses
# network.line['r_ohm_per_km'] = network.line['r_ohm_per_km'] * 5.0 #around 3%
network.line['r_ohm_per_km'] = network.line['r_ohm_per_km'] * 7.0 #around 4.5%

# desc = "Low Losses"
desc = "Medium Losses - Upscaled Line Resistance (Factor 5.0)"
# desc = "High Losses - Upscaled Line Resistance (Factor 7.0)"

#Note:
#Bus 39 represents interconnection to an aggregated New York system.
#The generator is therefore modelled with a very high inertia constant.


slack_gens = np.arange(0,10)
participation_factors = np.array([])
ref_bus_pset = 0 #undefined


#function loading test case system information and power flow results from PandaPower
system_ss = pf.load_pandapower_case(network, enforce_q_limits = True, distributed_slack = False, 
                                    slack_gens = slack_gens, participation_factors = participation_factors)[0]
system_ds = pf.load_pandapower_case(network, enforce_q_limits = True, distributed_slack = True, 
                                    slack_gens = slack_gens, participation_factors = participation_factors,
                                    ref_bus_pset = ref_bus_pset)[0]

#single slack bus on generator 0 (hydro power plant)
#assigning participation factor of 1.0
system_ss2 = pf.load_pandapower_case(network, enforce_q_limits = True, distributed_slack = True, 
                                    slack_gens = np.array([0]), participation_factors = np.array([1.0]),
                                    original_ref_bus=True)[0]

#loading proportional partcipation factors for distributed slack model
gens = system_ds.get('generators').copy()
if np.size(slack_gens) == 0:
    participation_factors = np.zeros(len(gens.index))
    slack_capacity = sum(gens['pmax'])
    for i in range(len(gens.index)):
        participation_factors[i] = gens['pmax'][i] / slack_capacity
else:
    gens_filt = gens.iloc[slack_gens].reset_index(drop=True)
    participation_factors = np.zeros(len(gens_filt.index))
    slack_capacity = sum(gens_filt['pmax'])
    for i in range(np.size(slack_gens)):
        participation_factors[i] = gens_filt['pmax'][i] / slack_capacity

#Downscaling slack weight of interconnection with high inertia constant
pi_rem = 0.60 * participation_factors[9]
participation_factors[9] *= 0.40

#Downscale slack weight of nuclear plants
for i in [1,2,5,7,8]:
    pi_rem += 0.25 * participation_factors[i]
    participation_factors[i] *= 0.75

#Distribute remaining participation across remaining generators
#based on proportional capacity
rem_gens = [0,3,4,6]
# rem_gens = [0,1,2,3,4,5,6,7,8]
for i in rem_gens:
    participation_factors[i] += (gens['pmax'][i] / sum(gens['pmax'][rem_gens])) * pi_rem


system_ds2 = pf.load_pandapower_case(network, enforce_q_limits = True, distributed_slack = True, 
                                    slack_gens = slack_gens, participation_factors = participation_factors,
                                    ref_bus_pset = ref_bus_pset)[0]


pf.new_england_case_line_fix(system_ss)
pf.new_england_case_line_fix(system_ss2)
pf.new_england_case_line_fix(system_ds)
pf.new_england_case_line_fix(system_ds2)


results_ss = pf.run_power_flow(system_ss, enforce_q_limits=True)
results_ss2 = pf.run_power_flow(system_ss2, enforce_q_limits=True)
results_ds = pf.run_power_flow(system_ds, enforce_q_limits=True)
results_ds2 = pf.run_power_flow(system_ds2, enforce_q_limits=True)


# pf.plot_results(system_ss, results_ss, angle=True, name = ('Case 1\n%s\nSingle Slack Bus: Base System') % desc)
# pf.plot_results(system_ss2, results_ss2, angle=True, name = ('Case 1\n%s\nSingle Slack Bus: Hydro Plant') % desc)
# pf.plot_results(system_ds, results_ds, angle=True, name = ('Case 1\n%s\nDistributed Slack Bus: Equal Participation Factors') % desc)
# pf.plot_results(system_ds2, results_ds2, angle=True, name = ('Case 1\n%s\nDistributed Slack Bus: Proportional Participation Factors') % desc)


# pf.plot_result_comparison(results_ss, results_ds, angle=True, name = ('Case 1\n%s\nSS vs. DS - Equal Participation Factors') % desc)

# name = ('Case 1\n%s\nSS vs. DS - Adjusted Participation Factors') % desc
pf.plot_result_comparison(results_ss, results_ds2, angle=True, fixed_y_axis_values=[0.005,20,20,10])

# pf.plot_result_comparison(results_ds, results_ds2, angle=True, name = ('Case 1\n%s\nDS - Equal vs. Adjusted Factors') % desc)

# name = ('Case 1\n%s\nSS - Original Slack vs. Hydro Plant Slack') % desc
# pf.plot_result_comparison(results_ss, results_ss2, angle=True, fixed_y_axis_values=[0.005,20,20,10])

# pf.plot_result_comparison(results_ds2, results_ss2, angle=True, name = ('Case 1\n%s\nSingle Hydro Plant Slack vs. DS Adjusted Factors') % desc)
