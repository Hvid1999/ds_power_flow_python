import power_flow_newton_raphson as pf
import pandapower.networks as nw
import numpy as np
import pandas as pd

pd.options.display.float_format = '{:.6f}'.format #avoid scientific notation when printing dataframes

#system tweaks
network = pf.new_england_39_new_voltages(nw.case39())
network.gen['vm_pu'][5] = 1.058

# 1: Load the various contingencies into multiple system dictionaries
# 2: Load results into multiple result dictionaries
# 3: Calculate super metric value
# 4: Otherwise proceed as for a single case
#Issue: Participation factors don't have the same size if number of slack gens vary



# Scaling line resistance to obtain more realistic system losses
# network.line['r_ohm_per_km'] = network.line['r_ohm_per_km'] * 3.5 #around 2%
network.line['r_ohm_per_km'] = network.line['r_ohm_per_km'] * 5.0 #around 3%
# network.line['r_ohm_per_km'] = network.line['r_ohm_per_km'] * 7.0

# desc = "Low Losses"
# desc = "Medium Losses - Upscaled Line Resistance (Factor 3.5)"
desc = "Medium Losses - Upscaled Line Resistance (Factor 5.0)"
# desc = "High Losses - Upscaled Line Resistance (Factor 7.0)"

#Note:
#Bus 39 represents interconnection to an aggregated New York system.
#The generator is therefore modelled with a very high inertia constant.

systems = []
base_systems = []
base_results = []
base_gens = []
ref_bus_pset = 0 #undefined

slack_gens = np.array([])
participation_factors = np.ones(10)
participation_factors = participation_factors / np.sum(participation_factors) #normalize
p_fact_initial = np.copy(participation_factors)

#==============================================================================
#Contingency 1 - loss of load
# slack_gens = np.arange(0,10)

systems.append(pf.load_pandapower_case(network, enforce_q_limits = True, distributed_slack = True, 
                                    slack_gens = slack_gens, participation_factors = participation_factors,
                                    ref_bus_pset = ref_bus_pset)[0])
base_systems.append(pf.load_pandapower_case(network, enforce_q_limits = True, distributed_slack = True, 
                                    slack_gens = slack_gens, participation_factors = participation_factors,
                                    ref_bus_pset = ref_bus_pset)[0])

pf.load_variation(systems[0], np.array([20]), scalings=np.ones(1)*0.0, const_pf=True)
pf.load_variation(base_systems[0], np.array([20]), scalings=np.ones(1)*0.0, const_pf=True)



#==============================================================================
#Contingency 2 - loss of generator bus
pf.panda_disable_bus(network, 31)


systems.append(pf.load_pandapower_case(network, enforce_q_limits = True, distributed_slack = True, 
                                    slack_gens = slack_gens, participation_factors = participation_factors,
                                    ref_bus_pset = ref_bus_pset)[0])
base_systems.append(pf.load_pandapower_case(network, enforce_q_limits = True, distributed_slack = True, 
                                    slack_gens = slack_gens, participation_factors = participation_factors,
                                    ref_bus_pset = ref_bus_pset)[0])


#==============================================================================
#Contingency 3 - loss of generator bus 
net = pf.new_england_39_new_voltages(nw.case39())
net.gen['vm_pu'][5] = 1.058
net.line['r_ohm_per_km'] = net.line['r_ohm_per_km'] * 3.5 #around 2%
# net.line['r_ohm_per_km'] = net.line['r_ohm_per_km'] * 5.0 #around 3%

pf.panda_disable_bus(net, 34)


systems.append(pf.load_pandapower_case(net, enforce_q_limits = True, distributed_slack = True, 
                                    slack_gens = slack_gens, participation_factors = participation_factors,
                                    ref_bus_pset = ref_bus_pset)[0])
base_systems.append(pf.load_pandapower_case(net, enforce_q_limits = True, distributed_slack = True, 
                                    slack_gens = slack_gens, participation_factors = participation_factors,
                                    ref_bus_pset = ref_bus_pset)[0])

#=============================================================================

for system in systems:
    pf.new_england_case_line_fix(system)

    system.update({'tolerance':1e-3})
    system.update({'iteration_limit':35})


for system in base_systems:
    pf.new_england_case_line_fix(system)
    base_gens.append(system.get('generators').copy())



gradient = np.ones(np.size(participation_factors))
epsilon = 1e-5
pf_count = 0
step_count = 0
gradient_old = np.copy(gradient)
p_fact_old = np.copy(participation_factors)

while (step_count < 20) and (np.linalg.norm(gradient) > 1e-2):
    results = []
    for system in systems:
        pf.load_participation_factors(system, participation_factors)
        results.append(pf.run_power_flow(system, enforce_q_limits=True, distributed_slack=True, 
                                         print_results=False))
    pf_count += 1
    print('\n%d...\n' % pf_count)
    
    phi = 0
    for result in results:
        phi += 0.975*pf.line_loading_metric(result) + 0.025*result.get('total_losses_pu') #combining metrics
        # phi += pf.line_loading_metric(result)
    phi = phi / len(results) #average metric over each contingency
    
    
    phi_pk = np.zeros(np.size(participation_factors))
    
    gradient_old = np.copy(gradient)
    
    for k in range(np.size(participation_factors)):
        p_fact_perturb = np.copy(participation_factors)
        p_fact_perturb[k] += epsilon #take small step
        p_fact_perturb = p_fact_perturb / np.sum(p_fact_perturb) #normalize
        
        #uncomment the line below to check PV bus reactive power every time (slower) instead of just the first time
        # system.update({'generators':gens_base.copy()})
        
        results = []
        for system in systems:
            pf.load_participation_factors(system, p_fact_perturb) #load new p-factors
            results.append(pf.run_power_flow(system, enforce_q_limits=True, distributed_slack=True, 
                                             print_results=False))
        
        #If a generator is inactive in a contingency, the metric perturbation of the corresponding 
        #participation factor is ignored
        ignore = 0
        for i in range(len(results)):
            # phi_pk[k] += pf.line_loading_metric(results[i])
            phi_pk[k] += 0.975*pf.line_loading_metric(results[i]) + 0.025*results[i].get('total_losses_pu') #combining metrics
            
        phi_pk[k] = phi_pk[k] / (len(results) - ignore) #averaging metric across contingencies
        
        gradient[k] = (phi_pk[k] - phi) / epsilon
        
        pf_count += 1
        print('\n%d...\n' % pf_count)
        
    if step_count == 0:
        gamma = 0.02 / np.max(gradient)
        phi_initial = phi
    else:
        #Barzilai-Borwein method for step size determination - Wikipedia
        #and https://arxiv.org/pdf/1907.06409.pdf
        gamma_BB = np.abs(np.vdot((participation_factors - p_fact_old), (gradient - gradient_old))) / (np.vdot(gradient - gradient_old, gradient - gradient_old))
        gamma_stab = 0.02 / np.linalg.norm(gradient)
        gamma = min(gamma_BB, gamma_stab)
    
    p_fact_old = np.copy(participation_factors)
    
    participation_factors = participation_factors - gamma * gradient #gradient step
    participation_factors = participation_factors / np.sum(participation_factors) #normalize
    
    step_count += 1
    
    if np.any(participation_factors < 0):
        #if a negative participation factor is obtained, set it to zero and renormalize
        #then break the loop
        neg_idx = np.where(participation_factors < 0)
        participation_factors[neg_idx] = 0
        participation_factors = participation_factors / np.sum(participation_factors) #normalize
        print('Sub-zero participation factor(s). Loop terminated.\n')
        break
    
    # system.update({'generators':gens_base.copy()})

print('\nFinished.\n')


for system in base_systems:
    base_results.append(pf.run_power_flow(system, enforce_q_limits=True, 
                                          distributed_slack=True, print_results=True))


pf.plot_results(base_systems[0], base_results[0], angle = True, singleplot = 'lines', name = ('Losing Load 20 - Equal Factors\n%s\nLosses: %f pu' % (desc, base_results[0].get('total_losses_pu'))))
pf.plot_results(systems[0], results[0], angle = True, singleplot = 'lines', name = ('Losing Load 20 - After Gradient Steps\n%s\nLosses: %f pu' % (desc, results[0].get('total_losses_pu'))))

pf.plot_results(base_systems[1], base_results[1], angle = True, singleplot = 'lines', name = ('Losing Bus 31 - Equal Factors\n%s\nLosses: %f pu' % (desc, base_results[1].get('total_losses_pu'))))
pf.plot_results(systems[1], results[1], angle = True, singleplot = 'lines', name = ('Losing Bus 31 - After Gradient Steps\n%s\nLosses: %f pu' % (desc, results[1].get('total_losses_pu'))))

pf.plot_results(base_systems[2], base_results[2], angle = True, singleplot = 'lines', name = ('Losing Bus 34 - Equal Factors\n%s\nLosses: %f pu' % (desc, base_results[2].get('total_losses_pu'))))
pf.plot_results(systems[2], results[2], angle = True, singleplot = 'lines', name = ('Losing Bus 34 - After Gradient Steps\n%s\nLosses: %f pu' % (desc, results[2].get('total_losses_pu'))))

# print("\nWarnings:\n")
# pf.check_p_limits(system, results)

