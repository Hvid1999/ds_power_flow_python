import ds_power_flow as pf
import pandapower.networks as nw
import numpy as np
import pandas as pd

pd.options.display.float_format = '{:.6f}'.format #avoid scientific notation when printing dataframes

#system tweaks
network = pf.new_england_39_new_voltages(nw.case39())
network.gen['vm_pu'][5] = 1.058

# Scaling line resistance to obtain more realistic system losses
network.line['r_ohm_per_km'] = network.line['r_ohm_per_km'] * 5.0 #around 3%
# network.line['r_ohm_per_km'] = network.line['r_ohm_per_km'] * 7.0

# desc = "Low Losses"
desc = "Medium Losses - Upscaled Line Resistance (Factor 5.0)"
# desc = "High Losses - Upscaled Line Resistance (Factor 7.0)"

slack_gens = np.arange(0,10)
# slack_gens = np.array([0,1,3,4,5,6,7,8,9])
# participation_factors = np.array([])
ref_bus_pset = 0 #undefined

participation_factors = np.ones(np.size(slack_gens))
# participation_factors = np.array([0.219, 0.066, 0.074, 0.137, 0.107, 0.070, 0.122, 0.057, 0.088, 0.060])
participation_factors = participation_factors / np.sum(participation_factors) #normalize
p_fact_initial = np.copy(participation_factors)

system = pf.load_pandapower_case(network, enforce_q_limits = True, distributed_slack = True, 
                                    slack_gens = slack_gens, participation_factors = participation_factors,
                                    ref_bus_pset = ref_bus_pset)[0]
pf.new_england_case_line_fix(system)


system.update({'tolerance':1e-3})
system.update({'iteration_limit':25})


system_base = pf.load_pandapower_case(network, enforce_q_limits = True, distributed_slack = True, 
                                    slack_gens = slack_gens, participation_factors = participation_factors,
                                    ref_bus_pset = ref_bus_pset)[0]
pf.new_england_case_line_fix(system_base)
gens_base = system_base.get('generators').copy()



gradient = np.ones(np.size(participation_factors))
epsilon = 1e-5
pf_count = 0
step_count = 0
gradient_old = np.copy(gradient)
p_fact_old = np.copy(participation_factors)



while (step_count < 20) and (np.linalg.norm(gradient) > 1e-2):
    results = pf.run_power_flow(system, enforce_q_limits=True, print_results=False)
    pf_count += 1
    print('\n%d...\n' % pf_count)
    phi = results.get('total_losses_pu')
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
        phi_pk[k] = results.get('total_losses_pu')
        
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
    
    system.update({'generators':gens_base.copy()})
    pf.load_participation_factors(system, participation_factors)

print('\nFinished.\nParticipation Factors:')
print(participation_factors)



results_base = pf.run_power_flow(system_base, enforce_q_limits=True, print_results=False)
phi_base = results_base.get('total_losses_pu')
pf.plot_results(system_base, results_base, angle = True, name = ('Equal Factors - Losses: %f\n%s' % (phi_base,desc)))
pf.plot_results(system, results, angle = True, name = ('After Gradient Steps - Losses: %f\n%s' % (phi,desc)))


#%%
#Randomized factors approach
ref_bus_pset = 0 #undefined


system = pf.load_pandapower_case(network, enforce_q_limits = True, distributed_slack = True, 
                                    slack_gens = np.array([]), participation_factors = np.array([]),
                                    ref_bus_pset = ref_bus_pset)[0]
pf.new_england_case_line_fix(system)

system.update({'tolerance':1e-3})
system.update({'iteration_limit':25})

results = pf.run_power_flow(system, enforce_q_limits=True, print_results=False)

equal_factors_loss = results.get('total_losses_pu')
min_loss = equal_factors_loss

num_attempts = 300

for n in range(1,num_attempts):  
    print(n)
    participation_factors = np.random.random(10)
    participation_factors = participation_factors / np.sum(participation_factors)
    pf.load_participation_factors(system, participation_factors)
    results = pf.run_power_flow(system, enforce_q_limits=True, print_results=False)
    
    if results.get('total_losses_pu') < min_loss:
        min_loss = results.get('total_losses_pu')
        best_factors = np.copy(participation_factors)
print('\nFinished.\nParticipation Factors:')
print(best_factors)
pf.plot_results(system, results, angle = True, name = ('After Randomized Search (n = %d) - Losses: %f\n%s' % (num_attempts,min_loss,desc)))
