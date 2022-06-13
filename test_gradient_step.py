import ds_power_flow as pf
import pandapower.networks as nw
import numpy as np
import pandas as pd
import time

pd.options.display.float_format = '{:.6f}'.format #avoid scientific notation when printing dataframes

#system tweaks
network = pf.new_england_39_new_voltages(nw.case39())
network.gen['vm_pu'][5] = 1.058

#Contingency
inactive_bus = 37
# pf.panda_disable_bus(network, inactive_bus)
pf.panda_disable_bus(network, inactive_bus)


# Scaling line resistance to obtain more realistic system losses
network.line['r_ohm_per_km'] = network.line['r_ohm_per_km'] * 4.8 #around 3%
# network.line['r_ohm_per_km'] = network.line['r_ohm_per_km'] * 7.0

# desc = "Low Losses"
desc = "Medium Losses - Upscaled Line Resistance (Factor 5.0)"
# desc = "High Losses - Upscaled Line Resistance (Factor 7.0)"


# slack_gens = np.arange(0,10)
slack_gens = np.array([])
# participation_factors = np.array([])
ref_bus_pset = 0 #undefined

participation_factors = np.ones(10)
# participation_factors = np.array([0.219, 0.066, 0.074, 0.137, 0.107, 0.070, 0.122, 0.057, 0.088, 0.060])
# participation_factors = np.delete(participation_factors, 2)
participation_factors = participation_factors / np.sum(participation_factors) #normalize
p_fact_initial = np.copy(participation_factors)

system = pf.load_pandapower_case(network, enforce_q_limits = True, distributed_slack = True, 
                                    slack_gens = slack_gens, participation_factors = participation_factors,
                                    ref_bus_pset = ref_bus_pset)[0]
pf.new_england_case_line_fix(system)


system.update({'tolerance':1e-3})
system.update({'iteration_limit':35})

system_base = pf.load_pandapower_case(network, enforce_q_limits = True, distributed_slack = True, 
                                    slack_gens = slack_gens, participation_factors = participation_factors,
                                    ref_bus_pset = ref_bus_pset)[0]
pf.new_england_case_line_fix(system_base)
gens_base = system_base.get('generators').copy()

for i in range(len(gens_base.index)):
    gens_base['pmax'][i] += 0.75  #increasing generator max real power for the sake of testing
system.update({'generators':gens_base})
system_base.update({'generators':gens_base})


gradient = np.ones(np.size(participation_factors))
epsilon = 1e-5
pf_count = 0
step_count = 0
step_count_limit = 20
gradient_old = np.copy(gradient)
p_fact_old = np.copy(participation_factors)
gamma_list = []
gamma_stab_list = []

t1 = time.time()
while (step_count < step_count_limit) and (np.linalg.norm(gradient) > 1e-2):
    results = pf.run_power_flow(system, enforce_q_limits=True, print_results=False)

    pf_count += 1
    print('\n%d...\n' % pf_count)
    # phi = 0.94*pf.line_loading_metric(results) + 0.06*pf.generator_limit_metric(system, results) #combining metrics
    # phi = 0.975*pf.line_loading_metric(results) + 0.025*results.get('total_losses_pu') #combining metrics
    # phi = pf.line_loading_metric(results)
    phi = pf.generator_limit_metric(system, results)
    phi_pk = np.zeros(np.size(participation_factors))
    
    gradient_old = np.copy(gradient)
    
    for k in range(np.size(participation_factors)):
        p_fact_perturb = np.copy(participation_factors)
        p_fact_perturb[k] += epsilon #take small step
        p_fact_perturb = p_fact_perturb / np.sum(p_fact_perturb) #normalize
        
        #uncomment the line below to check PV bus reactive power every time (slower) instead of just the first time
        # system.update({'generators':gens_base.copy()})
        pf.load_participation_factors(system, p_fact_perturb) #load new p-factors
        
        results = pf.run_power_flow(system, enforce_q_limits=True, print_results=False)
        # phi_pk[k] = 0.94*pf.line_loading_metric(results) + 0.06*pf.generator_limit_metric(system, results) #combining metrics
        # phi_pk[k] = 0.975*pf.line_loading_metric(results) + 0.025*results.get('total_losses_pu') #combining metrics
        # phi_pk[k] = pf.line_loading_metric(results)
        phi_pk[k] = pf.generator_limit_metric(system, results)
        
        
        #ignoring the effect of perturbing the factor of the inactive generator(s) in this single case
        if gens_base.in_service[k]:
            gradient[k] = (phi_pk[k] - phi) / epsilon
        else:
            gradient[k] = 0
            participation_factors[k] = 0
        
        pf_count += 1
        print('\n%d...\n' % pf_count)
        
    if step_count == 0:
        gamma = 0.02 / np.linalg.norm(gradient)
        phi_initial = phi
    else:
        #Barzilai-Borwein method for step size determination - Wikipedia
        #and https://arxiv.org/pdf/1907.06409.pdf
        gamma_BB = np.abs(np.vdot((participation_factors - p_fact_old), (gradient - gradient_old))) / (np.vdot(gradient - gradient_old, gradient - gradient_old))
        gamma_stab = 0.02 / np.linalg.norm(gradient)
        gamma = min(gamma_BB, gamma_stab)
        gamma_list.append(gamma_BB)
        gamma_stab_list.append(gamma_stab)
        
    
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
    
    system.update({'generators':gens_base.copy()})
    pf.load_participation_factors(system, participation_factors)

print('\nFinished.\nParticipation Factors:')
print(participation_factors)

t2 = time.time()
print('\nTime elapsed: %f s' % (t2-t1))

results_base = pf.run_power_flow(system_base, enforce_q_limits=True, print_results=False)


# name = ('Losing Bus %d - Equal Factors\n%s\nLosses: %f pu' % (inactive_bus,desc, results_base.get('total_losses_pu')))
pf.plot_results(system_base, results_base, angle = True, plot='lg', lg_lim=[115,0])
# name = ('Losing Bus %d - After Gradient Steps\n%s\nLosses: %f pu' % (inactive_bus,desc, results.get('total_losses_pu')))
pf.plot_results(system, results, angle = True, plot='lg',lg_lim=[115,110])

# print("\nWarnings:\n")
# pf.check_p_limits(system, results)
# pf.check_q_limits(system, results)
# pf.check_bus_voltage(system, results)
# pf.check_line_trafo_loading(system, results)