import numpy as np
import pandas as pd
import pandapower as pp
import matplotlib.pyplot as plt

def load_pandapower_case(network, enforce_q_limits=False, distributed_slack = False, 
                         slack_gens = np.array([]), participation_factors = np.array([])):
    baseMVA = network.sn_mva #base power for system
    freq = network.f_hz

    #run PandaPower power flow
    pp.runpp(network, enforce_q_lims = enforce_q_limits, trafo_model='pi', trafo_loading='power')
    #Saving PandaPower results and per-unitizing power values
    pandapower_results = network.res_bus
    pandapower_results['p_pu'] = pandapower_results.p_mw/baseMVA
    pandapower_results['q_pu'] = pandapower_results.q_mvar/baseMVA
    pandapower_results = pandapower_results[['vm_pu','va_degree','p_pu','q_pu']]

    ybus = network._ppc["internal"]["Ybus"].todense() #extract Ybus after running power flow
    gen = network.gen #voltage controlled generators
    sgen = network.sgen #static generators (PQ)
    load = network.load
    slack = network.ext_grid
    buses = network.bus #bus parameters
    lines = network.line #line parameters
    shunts = network.shunt #information about shunts
    trafo = network.trafo #transformers

    #Desired dataframe formats:

    # Generator (both sgen and gen): ['in_service', 'bus', 'type', 'vset', 'pset', 'qset', 'slack',
    #                                  'participation_factor', 'pmax', 'pmin', 'qmax', 'qmin']

    # Load: [in_service, bus, p, q]

    # Shunt: [in_service, bus, p, q]

    # Line: [in_service, from, to, parallel, length, c_nf_per_km, g_us_per_km, r_ohm_per_km, 
    #         x_ohm_per_km, ampacity]

    # Transformer: [in_service, lv_bus, hv_bus, parallel, s_rated, tap_pos, tap_min, tap_max, 
    #                tap_side, tap_step_percent]

    #Reformatting slack dataframe    
    slack = slack.rename(columns={'vm_pu':'vset', 'max_p_mw':'pmax', 'min_p_mw':'pmin',
                                  'max_q_mvar':'qmax', 'min_q_mvar':'qmin'})
    slack = slack[['in_service', 'bus', 'vset', 'pmax', 'pmin', 'qmax','qmin']]
    slack[['pmax', 'pmin', 'qmax', 'qmin']] = slack[['pmax', 'pmin',
                                                                   'qmax', 'qmin']] / baseMVA

    #Reformatting generator dataframe
    gen['type'] = 'pv'
    gen['qset'] = None
    gen = gen.rename(columns = {'p_mw':'pset', 'max_p_mw':'pmax', 'min_p_mw':'pmin',
                                  'max_q_mvar':'qmax', 'min_q_mvar':'qmin', 'vm_pu':'vset'})
    gen = gen[['in_service', 'bus', 'type', 'vset', 'pset', 'qset', 'pmax', 'pmin', 'qmax', 'qmin']]
    #Per-unitizing values
    gen[['pset', 'pmax', 'pmin', 'qmax', 'qmin']] = gen[['pset', 'pmax', 'pmin',
                                                                   'qmax', 'qmin']] / baseMVA

    if len(sgen.index) == 0:
        gens = gen #if there are no static generators
    else:
        #adding static generators to generator list
        sgen['vset'] = None
        sgen['type'] = 'pq'
        sgen = sgen.rename(columns = {'p_mw':'pset', 'q_mvar':'qset', 'max_p_mw':'pmax', 'min_p_mw':'pmin',
                                      'max_q_mvar':'qmax', 'min_q_mvar':'qmin'})

        sgen = sgen[['in_service', 'bus', 'type', 'vset', 'pset', 'qset', 'pmax', 'pmin', 'qmax', 'qmin']]
        #Per-unitizing values
        sgen[['pset', 'qset', 'pmax', 'pmin', 'qmax', 'qmin']] = sgen[['pset', 'qset', 'pmax', 'pmin',
                                                                       'qmax', 'qmin']] / baseMVA
        gens = pd.concat([gen,sgen]) 


    gens = gens.sort_values(by=['bus'])
    gens = gens.reset_index(drop=True)

    #Reformatting load and shunt dataframes
    load = load.rename(columns={'p_mw':'p', 'q_mvar':'q'})
    load = load[['in_service','bus', 'p', 'q']]
    load[['p','q']] = load[['p','q']] / baseMVA

    shunts = shunts.rename(columns={'p_mw':'p', 'q_mvar':'q'})
    shunts = shunts[['in_service','bus', 'p', 'q']]
    shunts[['p','q']] = shunts[['p','q']] / baseMVA

    #Note! Shunts are basically handled as loads in this code.
    #PandaPower implements shunts in a different way - they essentially do not 
    #affect bus voltages, but instead their power consumption is calculated 
    #and added to the bus power consumption
    #more on this: https://pandapower.readthedocs.io/en/v2.8.0/elements/shunt.html 


    #Reformatting lines dataframe
    lines = lines.rename(columns={'from_bus':'from', 'to_bus':'to', 'length_km':'length',
                                  'max_i_ka':'ampacity_ka'})
    lines = lines[['in_service','from','to','parallel','length','c_nf_per_km', 'g_us_per_km', 'r_ohm_per_km',
                   'x_ohm_per_km','ampacity_ka']]

    #Reformatting transformers dataframe
    trafo = trafo.rename(columns={'sn_mva':'s_rated'})
    trafo = trafo[['in_service', 'lv_bus', 'hv_bus', 'parallel', 's_rated', 'tap_pos', 'tap_min', 'tap_max',
                   'tap_side', 'tap_step_percent']]

    buses = buses[['in_service','vn_kv', 'max_vm_pu', 'min_vm_pu','name']]
    
    #loading power factors for loads
    load['pf'] = load['p']/(np.sqrt(load['p']**2 + load['q']**2))
    
    #Setup system dictionary
    system = {'n_buses':ybus.shape[0],'distributed_slack':distributed_slack, 'admmat':ybus,'slack':slack,
              'iteration_limit':15,'tolerance':1e-3, 's_base':baseMVA, 'frequency':freq}
    system.update({'generators':gen})
    system.update({'loads':load})
    system.update({'shunts':shunts})
    system.update({'buses':buses})
    system.update({'lines':lines})
    system.update({'transformers':trafo})


    if distributed_slack:
        system.update({'reference_bus':system.get('slack').bus[0]}) #saving original single slack bus
        del system['slack'] #removing separate slack bus description
        
        #The setpoint original slack bus generator is difference between total load and total generation
        load_sum = 0
        gen_sum = 0

        for i in range(len(load.index)):
            load_sum += load.p[i]

        for i in range(len(gens.index)):
            gen_sum += gens.pset[i]

        slack_pset = load_sum - gen_sum

        #Adding original slack bus generator as PV-bus generator
        slack_to_gen = {'in_service':slack.in_service[0], 'bus':slack.bus[0], 'type':'pv', 'vset':slack.vset[0], 
                        'pset':slack_pset, 'qset':None, 'pmax':slack.pmax[0], 
                        'pmin':slack.pmin[0], 'qmax':slack.qmax[0], 
                        'qmin':slack.qmin[0]}

        #Adding slack bus to generator dataframe and re-sorting by bus
        gens = gens.append(slack_to_gen, ignore_index = True)
        gens = gens.sort_values(by=['bus'])
        gens = gens.reset_index(drop=True)
        
        gens['slack'] = True #if no specific slack generators are entered, every generator participates
        gens['participation_factor'] = 0.0
        
        if np.size(slack_gens) != 0: 
            for i in range(len((gens.index))):
                if i in slack_gens:
                    gens.slack[i] = True
                else: 
                    gens.slack[i] = False
                    
        system.update({'generators':gens})
        
        load_participation_factors(system, p_factors=participation_factors) #loading either equal p-factors or custom ones

    return (system, pandapower_results)



def process_admittance_mat(system):
    ybus = system.get('admmat')
    n_buses = ybus.shape[0]
    g = np.real(ybus)
    b = np.imag(ybus)
    return n_buses, g, b

def get_pv_idx(system):
    
    pv_idx = np.empty((1,0), dtype=int)
    gens = system.get('generators')
    for i in range(len(gens.index)):
        if gens.type[i] == 'pv':
            pv_idx = np.append(pv_idx, gens.bus[i])
    
    return pv_idx

def slack_idx(system):
    if system.get('distributed_slack'):
        return system.get('reference_bus')
    else:
        return system.get('slack').bus[0]

def init_voltage_vecs(system):
    n_buses = system.get('n_buses')
    #vectors containing voltage magnitude and angle information on all busses
    vmag_full = np.ones((n_buses,1))
    delta_full = np.zeros((n_buses,1))

    #Checking for PV-busses in order to simplify eventual calculations
    pv_idx = get_pv_idx(system)
    vset = np.empty((1,0), dtype=int)
    gens = system.get('generators')
    
    #loading voltage setpoints for PV generators
    for i in range(len(gens.index)):
        if gens.type[i] == 'pv':
            vset = np.append(vset, gens.vset[i])
    vset = np.reshape(vset, (np.size(vset),1))
            
    if np.size(pv_idx) != 0:
        vmag_full[pv_idx] = vset
    
    if system.get('distributed_slack'):
        vmag = np.delete(vmag_full, pv_idx, 0) #removing known PV bus voltage magnitudes
    else:
        #setting slack bus voltage magnitude
        vmag_full[slack_idx(system)] = system.get('slack').vset[0]
        pv_slack_idx = np.sort(np.append(pv_idx, slack_idx(system))) #pv and slack indices
        vmag = np.delete(vmag_full, pv_slack_idx, 0) #removing slack bus and PV busses
    
    delta = np.delete(delta_full, slack_idx(system), 0)#reference voltage angle
    
    #removal of slack bus index from non-full vectors in return statement
    return vmag, delta, vmag_full, delta_full


def calc_power_vecs(system, vmag, delta, g, b):
    ybus = system.get('admmat')
    n_buses = ybus.shape[0]
    
    #vectors with possibility for containing information about every bus
    p_full = np.zeros((n_buses,1))
    q_full = np.zeros((n_buses,1))
    
    for k in range(n_buses): 
        psum = 0
        qsum = 0
        for n in range(n_buses):
            psum += vmag[n] * (g[k,n]*(np.cos(delta[k] - delta[n])) + b[k,n]*np.sin(delta[k] - delta[n]))
            qsum += vmag[n] * (g[k,n]*(np.sin(delta[k] - delta[n])) - b[k,n]*np.cos(delta[k] - delta[n]))
        p_full[k] = vmag[k] * psum
        q_full[k] = vmag[k] * qsum

    #Checking for PV-busses in order to simplify eventual calculations
    pv_idx = get_pv_idx(system)
    
    
    if system.get('distributed_slack'):
        q = np.delete(q_full, pv_idx, 0) #removing the pv bus indices after calculation
        p = p_full
    else:
        pv_slack_idx = np.sort(np.append(pv_idx, slack_idx(system))) #pv and slack indices
        q = np.delete(q_full, pv_slack_idx, 0) #removing the pv and slack bus indices after calculation
        p = np.delete(p_full, slack_idx(system), 0) 
        
    return p, q, p_full, q_full


def calc_power_setpoints(system):
    n_buses = system.get('n_buses')
    
    #loading bus setpoints
    pset = np.zeros((n_buses,1))
    qset = np.zeros((n_buses,1))
    
    gens = system.get('generators')
    loads = system.get('loads')
    shunts = system.get('shunts')
    
    for i in range(len(loads.index)):
        if loads.in_service[i]:
            k = loads.bus[i]
            pset[k] -= loads.p[i] #load is a negative injection
            qset[k] -= loads.q[i]
    
    #Shunts are as of now handled as loads 
    #this allows them to affect bus voltages - contrary to PandaPower shunts
    
    for i in range(len(shunts.index)):
        if shunts.in_service[i]:
            k = shunts.bus[i]
            pset[k] -= shunts.p[i] #shunt values are consumption (load convention)
            qset[k] -= shunts.q[i]
    
        
    for i in range(len(gens.index)):
        if gens.in_service[i]:
            k = gens.bus[i]
            pset[k] += gens.pset[i] #generator is a positive injection
            if gens.type[i] == 'pq':
                qset[k] += gens.qset[i]
    
    pv_idx = get_pv_idx(system)
    
    
    if system.get('distributed_slack'):
        qset = np.delete(qset, pv_idx, 0)
    else:
        pv_slack_idx = np.sort(np.append(pv_idx, slack_idx(system))) #pv and slack indices
        qset = np.delete(qset, pv_slack_idx, 0) #removing PV and slack bus indices
        pset = np.delete(pset, slack_idx(system), 0) #removing slack bus index
    
    return pset, qset


def calc_mismatch_vecs(system, p, q):
    
    (pset, qset) = calc_power_setpoints(system)
    
    del_p = pset - p
    del_q = qset - q
    return del_p, del_q


def calc_jacobian(system, vmag, delta, g, b, p, q):
    n_buses = system.get('n_buses')
    
    jacobian = np.zeros((2*(n_buses),2*(n_buses)))
    
    #Pointing to the submatrices
    j1 = jacobian[0:(n_buses),0:(n_buses)]
    j2 = jacobian[0:(n_buses),(n_buses):(2*(n_buses))]
    j3 = jacobian[(n_buses):(2*(n_buses)),0:(n_buses)]
    j4 = jacobian[(n_buses):(2*(n_buses)),(n_buses):(2*(n_buses))]

    #Calculating Jacobian matrix
    for k in range(n_buses):
        for n in range(n_buses):
            if k == n: #diagonal elements
                j1[k,n] = -q[k] - b[k,k] * vmag[k]**2
                j2[k,n] = p[k] / vmag[k] + g[k,k] * vmag[k]
                j3[k,n] = p[k] - g[k,k] * vmag[k]**2
                j4[k,n] = q[k] / vmag[k] - b[k,k] * vmag[k]

            else: #off-diagonal elements
                j1[k,n] = vmag[k] * vmag[n] * (g[k,n]*(np.sin(delta[k] - delta[n])) - b[k,n]*np.cos(delta[k] - delta[n]))
                j2[k,n] = vmag[k] * (g[k,n]*(np.cos(delta[k] - delta[n])) + b[k,n]*np.sin(delta[k] - delta[n]))
                j3[k,n] = -vmag[k] * vmag[n] * (g[k,n]*(np.cos(delta[k] - delta[n])) + b[k,n]*np.sin(delta[k] - delta[n]))
                j4[k,n] = vmag[k] * (g[k,n]*(np.sin(delta[k] - delta[n])) - b[k,n]*np.cos(delta[k] - delta[n]))
    
    if system.get('distributed_slack'):
        #loading vector of participation factors from the system
        part_facts = np.zeros((2 * n_buses, 1)) #includes the zero vector below p-factor vector
        gens = system.get('generators')
        
        for i in range(len(gens.index)):
            if gens.slack[i]:
                k = gens.bus[i]
                part_facts[k] = gens.participation_factor[i]
        
        jacobian = np.append(jacobian, part_facts, axis = 1)
    
    return jacobian

def jacobian_calc_simplify(system, jacobian):
    n_buses = system.get('n_buses')
    pv_idx = get_pv_idx(system) #reading indices of PV-busses
    row_remove = np.array([], dtype=int)
    col_remove = np.array([], dtype=int)

    if system.get('distributed_slack'):
        ref_idx = system.get('reference_bus')
        #voltage angle is assumed known at the reference bus
        #the corresponding column of J1 and J3 may therefore be removed
        col_remove = np.append(col_remove, ref_idx)
        col_remove = np.append(col_remove, pv_idx + n_buses) #offset of n_buses to reach J2 and J4
        
        row_remove = np.append(row_remove, pv_idx + n_buses) #offset of n_buses to reach J3 and J4
        
    else:
        slack_index = slack_idx(system)
        col_remove = np.append(col_remove, slack_index)
        col_remove = np.append(col_remove, slack_index + n_buses)
        row_remove = np.append(row_remove, slack_index)
        row_remove = np.append(row_remove, slack_index + n_buses)
        
        #PV bus simplification
        col_remove = np.append(col_remove, pv_idx + n_buses) #offset of n_buses to reach J2 and J4
        row_remove = np.append(row_remove, pv_idx + n_buses) #offset of n_buses to reach J3 and J4
    
    #Deleting relevant rows and columns    
    jacobian_calc = np.delete(jacobian, row_remove, 0) 
    jacobian_calc = np.delete(jacobian_calc, col_remove, 1)
    
    return jacobian_calc


def next_iteration(jacobian, vmag, delta, del_p, del_q):
    x = np.row_stack((delta, vmag))
    y = np.row_stack((del_p, del_q))
    x_next = x + np.matmul(np.linalg.inv(jacobian), y) #calculating next iteration
    delta_next = x_next[0:np.size(delta)]
    vmag_next = x_next[np.size(delta):]
    return delta_next, vmag_next


def dist_next_iteration(jacobian, vmag, delta, k_g, del_p, del_q):
    
    x = np.row_stack((delta, vmag))
    x = np.append(x, [[k_g]], axis = 0)

    y = np.row_stack((del_p, del_q))

    x_next = x + np.matmul(np.linalg.inv(jacobian), y) #calculating next iteration
    
    delta_next = x_next[0:np.size(delta)]
    vmag_next = x_next[np.size(delta):(np.size(x_next) - 1)]
    k_g_next = x_next[-1][0]
    
    return delta_next, vmag_next, k_g_next


def check_convergence(delta_next, vmag_next, delta, vmag, threshold):
    #returns true if all indices in mismatch vector are below error threshold (tolerance)
    #based on magnitude in change of iteration values for voltages
    x_next = np.row_stack((delta_next, vmag_next))
    x = np.row_stack((delta, vmag))
    checkvec = np.ones((x.shape))
    
    for i in range(np.size(x)):
        if abs(x[i]) > 0: #avoid division by zero
            checkvec[i] = (x_next[i] - x[i])/x[i]
    
    return np.all(np.absolute(checkvec) < threshold) 

def dist_check_convergence(delta_next, vmag_next, delta, vmag, k_g_next, k_g, threshold):
    #returns true if all indices in mismatch vector are below error threshold (tolerance)
    #based on magnitude in change of iteration values for voltages
    x_next = np.row_stack((delta_next, vmag_next))
    x_next = np.append(x_next, [[k_g_next]], axis = 0)
    x = np.row_stack((delta, vmag))
    x = np.append(x, [[k_g]], axis = 0)
    checkvec = np.ones((x.shape))
    
    for i in range(np.size(x)):
        if abs(x[i]) > 0: #avoid division by zero
            checkvec[i] = (x_next[i] - x[i])/x[i]
    
    return np.all(np.absolute(checkvec) < threshold) 


def check_convergence_old(y, threshold):
    #returns true if all indices in mismatch vector are below error threshold (tolerance)
    #based entirely on power mismatches
    return np.all(np.absolute(y) < threshold) 


def check_pv_bus(system, n_buses, q_full):
    #check if PV bus reactive power is within specified limits
    #if not, set bus(es) to PQ at Q limit and return a bool to specify whether recalculation should be performed
    limit_violation = False
    
    
    #Only the generator outputs should be considered, so the generator loads must be subtracted
    #when checking the limit violation for reactive power!    
    q_loads = np.zeros((n_buses,1))
    
    loads = system.get('loads')
    gens = system.get('generators')
    shunts = system.get('shunts')
    
    
    for i in range(len(loads.index)):
        k = loads.bus[i]
        q_loads[k] -= loads.q[i]
        
    for i in range(len(shunts.index)):
        k = shunts.bus[i]
        q_loads[k] -= shunts.q[i]
    
    #Note: This should perhaps account for any sgens present on the same bus
    #However, not relevant for the New England 39-bus system
    
    for i in range(len(gens.index)):
        if (gens.type[i] == 'pv') and (gens.in_service[i]): #only considering in service PV-busses
            k = gens.bus[i]
            if (q_full[k] - q_loads[k]) < gens.qmin[i]:
                qset = gens.qmin[i]
                gens.qset[i] = qset
                gens.type[i] = 'pq'
                limit_violation = True
                break
            elif (q_full[k] - q_loads[k]) > gens.qmax[i]:
                qset = gens.qmax[i]
                gens.qset[i] = qset
                gens.type[i] = 'pq'
                limit_violation = True
                break
            system.update({'generators':gens})
    
    if limit_violation == True:
        print('Generator reactive power limit violated at bus %d.\nType set to PQ with generator reactive power setpoint of %.2f pu.\n' % (k, qset))
    
    return limit_violation


def check_p_limits(system, p):
    #calculate vector of generator outputs (p - p_load)
    
    #for systems with multiple generators on a single bus (typically static generators)
    #compare sum of generator outputs to sum of maximum power outputs
    p_gen = np.copy(p)
    
    gens = system.get('generators')
    loads = system.get('loads')
    # p_gen = np.zeros((system.get('n_buses'),1))
    #p_load = np.copy(p_gen)
    p_limits = np.zeros((system.get('n_buses'),1))
    gen_idx = np.array([], dtype=int)

    #Calculating vector of generator outputs

    for i in range(len(loads.index)):
        k = loads.bus[i]
        p_gen[k] += loads.p[i] #removing the negative load injections from the power vector

    for i in range(len(gens.index)):
        k = gens.bus[i]
        gen_idx = np.append(gen_idx, k)
        if not (np.isnan(gens.pmax[i]) or (gens.pmax[i] is None)): #if it is not nan or None
            p_limits[k] += gens.pmax[i]

    for k in np.unique(gen_idx):
        if p_gen[k] > p_limits[k]:
            magnitude = p_gen[k] - p_limits[k]
            print("\nGenerator(s) real power limit(s) exceeded at bus %i by %f pu.\n" 
                  % (k, magnitude))
            
    return


def check_q_limits(system, q):
    #Only relevant if reactive power limits are not enforced in the power flow
    
    #for systems with multiple generators on a single bus (typically static generators)
    #compare sum of generator outputs to sum of maximum power outputs
    q_gen = np.copy(q)
    
    gens = system.get('generators')
    loads = system.get('loads')
    shunts = system.get('shunts')
    # p_gen = np.zeros((system.get('n_buses'),1))
    #p_load = np.copy(p_gen)
    q_max = np.zeros((system.get('n_buses'),1))
    q_min = np.zeros((system.get('n_buses'),1))
    gen_idx = np.array([], dtype=int)

    #Calculating vector of generator outputs

    for i in range(len(loads.index)):
        k = loads.bus[i]
        q_gen[k] += loads.q[i] #removing the negative load injections from the power vector
    
    for i in range(len(shunts.index)):
        k = shunts.bus[i]
        q_gen[k] += shunts.q[i] 

    for i in range(len(gens.index)):
        k = gens.bus[i]
        gen_idx = np.append(gen_idx, k)
        if not (np.isnan(gens.qmax[i]) or (gens.qmax[i] is None)): #if it is not nan or None
            q_max[k] += gens.qmax[i]
            
        if not (np.isnan(gens.qmin[i]) or (gens.qmin[i] is None)):
            q_min[k] += gens.qmin[i]

    for k in np.unique(gen_idx):
        if q_gen[k] > q_max[k]:
            magnitude = q_gen[k] - q_max[k]
            print("\nGenerator(s) reactive power upper limit(s) exceeded at bus %i by %f pu.\n" 
                  % (k, magnitude))
        elif q_gen[k] < q_min[k]:
            magnitude = abs(q_min[k] - q_gen[k])
            print("\nGenerator(s) reactive power lower limit(s) exceeded at bus %i by %f pu.\n" 
                  % (k, magnitude))
            
    return

def check_bus_voltage(system, vmag):
    bus = system.get('buses')
    
    for i in range(len(bus.index)):
        if vmag[i] > bus.max_vm_pu[i]:
            magnitude = vmag[i] - bus.max_vm_pu[i]
            print("\nBus voltage upper limit exceeded at bus %i by %f pu.\n" 
                  % (i, magnitude))
        elif vmag[i] < bus.min_vm_pu[i]:
            magnitude = bus.min_vm_pu[i] - vmag[i]
            print("\nBus voltage lower limit exceeded at bus %i by %f pu.\n" 
                  % (i, magnitude))
        elif (abs(vmag[i] - bus.min_vm_pu[i]) < 0.005) or (abs(vmag[i] - bus.max_vm_pu[i]) < 0.005):
            print("\nBus voltage near limit at bus %i (%f pu).\n" % (i, vmag[i]))
    return

def check_line_trafo_loading(system, results):
    lines = system.get('lines')
    trafo = system.get('transformers')
    l_flows = results.get('line_flows')
    t_flows = results.get('transformer_flows')
    
    for i in range(len(l_flows.index)):
        if l_flows.loading_percent[i] > 100:
            f = lines['from'][i]
            t = lines['to'][i]
            print("\nLine %i (bus %i to %i) overloaded at %f %%.\n" 
                  % (i, f, t, l_flows.loading_percent[i]))
        elif (100 - l_flows.loading_percent[i]) < 5:
            f = lines['from'][i]
            t = lines['to'][i]
            print("\nLine %i (bus %i to %i) near limit at %f %%.\n" 
                  % (i, f, t, l_flows.loading_percent[i]))
    
    for i in range(len(t_flows.index)):
        if t_flows.loading_percent[i] > 100:
            lv = trafo.lv_bus[i]
            hv = trafo.hv_bus[i]
            print("\nTransformer %i (bus %i to %i) overloaded at %f %%.\n" 
                  % (i, lv, hv, t_flows.loading_percent[i]))
        elif (100 - t_flows.loading_percent[i]) < 5:
            lv = trafo.lv_bus[i]
            hv = trafo.hv_bus[i]
            print("\nTransformer %i (bus %i to %i) near limit at %f %%.\n" 
                  % (i, lv, hv, t_flows.loading_percent[i]))
    
    return


def get_phasor(mags, args, idx):
    return complex(mags[idx]*np.cos(args[idx]), mags[idx]*np.sin(args[idx]))
    


def calc_line_flows(system, vmag, delta):
    #Line flows: Current, real, reactive, apparent power at each end of lines
    #P_ft, Ptf, Q_ft, Q_tf, I_ft, I_tf, S_ft, S_tf
    #where ft = from/to and tf = to/from
    
    
    s_base = system.get('s_base')
    freq = system.get('frequency')
    
    lines = system.get('lines')
    n_lines = len(lines.index)

    #initializing empty arrays for storing data
    i_ft_pu = np.zeros(n_lines, dtype = complex)
    i_tf_pu = np.zeros(n_lines, dtype = complex)

    s_ft_pu = np.zeros(n_lines, dtype = complex)
    s_tf_pu = np.zeros(n_lines, dtype = complex)

    i_ka = np.zeros(n_lines)
    loading_percent = np.zeros(n_lines)


    for i in range(n_lines):
        if lines.in_service[i]:
            l = lines.length[i]
            parallel = lines.parallel[i] #number of lines in parallel
            from_idx = lines['from'][i]
            to_idx = lines['to'][i]
            
            #relevant base values for per unit calculations
            v_base = system.get('buses').vn_kv[from_idx]
            z_base = (v_base ** 2)  / (s_base) #voltage in kV and power in MVA
            # I_base = S_base_3ph / sqrt(3) * V_base_LL
            i_base_ka = s_base * 1e3 / (np.sqrt(3) * v_base * 1e3) #base current in kA (power base multiplied by 1e3 instead of 1e6)
            
            
            y_shunt = complex(lines['g_us_per_km'][i] * 1e-6, 
                              2 * np.pi * freq * lines['c_nf_per_km'][i]*1e-9) * l * parallel
            y_shunt_pu =  y_shunt * z_base # Y = 1/Z, so Y_pu = 1/Z_pu = Y * Z_base
            
            z_line = complex(lines['r_ohm_per_km'][i], lines['x_ohm_per_km'][i]) * l / parallel
            z_line_pu = z_line / z_base
            
            #loading voltage magnitude and phase angle as phasor
            v_1 = get_phasor(vmag, delta, from_idx)
            v_2 = get_phasor(vmag, delta, to_idx)
            
            # I_12 = (V_1 - V_2) / (Z_12) + V_1 / Y_sh / 2
            
            i_ft_pu[i] = ((v_1 - v_2) / z_line_pu + v_1 * (y_shunt_pu / 2))
            i_tf_pu[i] = ((v_2 - v_1) / z_line_pu + v_2 * (y_shunt_pu / 2))
            
            s_ft_pu[i] = v_1 * np.conj(i_ft_pu[i])
            s_tf_pu[i] = v_2 * np.conj(i_tf_pu[i])
            
            i_ka[i] = max(np.abs(i_ft_pu[i]), np.abs(i_tf_pu[i])) * i_base_ka
            
            loading_percent[i] = (i_ka[i] / lines['ampacity_ka'][i]) * 100
        
        
    p_ft_pu = np.real(s_ft_pu)
    p_tf_pu = np.real(s_tf_pu)

    q_ft_pu = np.imag(s_ft_pu)
    q_tf_pu = np.imag(s_tf_pu)

    p_loss = p_ft_pu + p_tf_pu
    q_loss = q_ft_pu + q_tf_pu
        
    d = {'loading_percent':loading_percent, 'i_ka':i_ka, 'p_ft_pu':p_ft_pu, 'p_tf_pu':p_tf_pu, 
         'p_loss_pu':p_loss, 'q_ft_pu':q_ft_pu, 'q_tf_pu':q_tf_pu, 'q_loss_pu':q_loss, 
         'i_ft_pu':np.abs(i_ft_pu), 'i_tf_pu':np.abs(i_tf_pu), 's_ft_pu':np.abs(s_ft_pu), 
         's_tf_pu':np.abs(s_tf_pu)}
    df = pd.DataFrame(data=d, index = np.arange(n_lines))
    df.index.name = 'line'
    
    return df


def calc_transformer_loadings(system, vmag, delta):
    #Note: Simplified representation of transformer as a series impedance between busses
    #(typical per-unit representation)

    
    trafo = system.get('transformers')
    ybus = system.get('admmat')
    s_base = system.get('s_base')
    n_trafo = len(trafo.index)
    
    #initializing empty arrays for storing data
    i_lv_pu = np.zeros(n_trafo, dtype = complex)
    i_hv_pu = np.zeros(n_trafo, dtype = complex)

    s_lv_pu = np.zeros(n_trafo, dtype = complex)
    s_hv_pu = np.zeros(n_trafo, dtype = complex)
    
    loading_percent = np.zeros(n_trafo)
    
    for i in range(n_trafo):
        lv = trafo.lv_bus[i]
        hv = trafo.hv_bus[i]
        v_lv = get_phasor(vmag, delta, lv)
        v_hv = get_phasor(vmag, delta, hv)
        x_t = 1 / ybus[lv, hv] #loading the series impedance from the admittance matrix
        
        i_lv_pu[i] = (v_lv - v_hv) / x_t
        i_hv_pu[i] = (v_hv - v_lv) / x_t
        
        s_lv_pu[i] = v_lv * np.conj(i_lv_pu[i])
        s_hv_pu[i] = v_hv * np.conj(i_hv_pu[i])
        
        s_mva = abs(max(s_lv_pu[i], s_hv_pu[i]) * s_base)
        
        loading_percent[i] = (s_mva / trafo['s_rated'][i]) * 100
    
    d = {'loading_percent':loading_percent, 'p_lv_pu':np.real(s_lv_pu), 'p_hv_pu':np.real(s_hv_pu), 
         'p_loss_pu':np.real(s_lv_pu) + np.real(s_hv_pu), 'q_lv_pu':np.imag(s_lv_pu), 
         'q_hv_pu':np.imag(s_hv_pu), 'q_loss_pu':np.imag(s_lv_pu) + np.imag(s_hv_pu), 
         'i_lv_pu':np.abs(i_lv_pu), 'i_hv_pu':np.abs(i_hv_pu), 's_lv_pu':np.abs(s_lv_pu), 
         's_hv_pu':np.abs(s_hv_pu)}
    df = pd.DataFrame(data=d, index = np.arange(n_trafo))
    df.index.name = 'trafo'
    
    return df


def calc_system_losses(system, vmag, delta):
    #Computes the system real power losses based on the loss function
    #especially relevant for distributed slack
    (n_buses, g, b) = process_admittance_mat(system)
    losses = 0

    for k in range(n_buses):
        losses += vmag[k] ** 2 * g[k,k]
        for n in range(k + 1, n_buses): #starts at n = k + 1 to avoid n == k as well as repeating behavior
            losses += 2 * vmag[k] * vmag[n] * g[k,n] * np.cos(delta[k] - delta[n])
            
    return losses


def load_participation_factors(system, p_factors = np.array([])):
    #accepts an array of participation factors ordered by increasing generator bus indices
    #if no array is entered, slack is distributed evenly among generators participating in slack
    gens = system.get('generators')
    
    slack_gens = gens[gens.slack & gens.in_service] 
    
    num_slack = len(slack_gens.index)
    
    if np.size(p_factors) == 0: #standard case for no input
        participation_factors = np.ones(num_slack)
        participation_factors = participation_factors / num_slack
    elif (np.size(p_factors) == num_slack) and (round(sum(p_factors),3) == 1.0):
        #the size of the p-factor vector must be the number of slack generators
        #the sum of the p-factors must be 1
        participation_factors = p_factors
    else:
        print('Error loading participation factors (sum = %f) - check input.' % sum(p_factors))
        print('Set to equal factors (standard case).\n')
        participation_factors = np.ones(num_slack)
        participation_factors = participation_factors / num_slack
    
    j = 0
    
    for i in range(len(gens.index)):
        if gens.slack[i]:
            if gens.in_service[i]:
                gens.participation_factor[i] = participation_factors[j]
                j += 1
            else:
                gens.participation_factor[i] = 0.0 #if the generator is disabled
                
    system.update({'generators':gens})
    return

def slack_distribution(system, k_g):
    
    gens = system.get('generators')
    slackvec = np.zeros((system.get('n_buses'), 1))
    
    for i in range(len(gens.index)):
        if gens.slack[i]:
            k = gens.bus[i]
            p_fact = gens.participation_factor[i]
            #k_g is a negative injection, but the absolute value is taken here
            #because the vector denotes how much each slack generator injects
            #to compensate for losses
            slackvec[k] = p_fact * k_g
    
    return slackvec


def load_variation(system, load_indices, scalings):
    #accepts an array of load indices to scale and an array of the 
    #corresponding scaling factors
    
    loads = system.get('loads')
    gens = system.get('generators')
    j = 0
    psi_load = 0
    
    for i in load_indices:
        p_old = loads.p[i]
        p_new = p_old * scalings[j]
        
        q_old = loads.q[i]
        q_new = q_old * scalings[j]
        
        psi_load += p_new - p_old
        
        loads.p[i] = p_new
        loads.q[i] = q_new
        
        j += 1
        print("\nLoad at bus %i changed from %f to %f (real power)\nAnd %f to %f (reactive power)." % (loads.bus[i], p_old, p_new, q_old, q_new))
    
    if system.get('distributed_slack'):
        print("\nTotal change in real power load: %f pu" % psi_load)
        print("Distribution across participating slack generators:")
        slackvec = slack_distribution(system, psi_load)
        slack_distribution_df = pd.DataFrame(data={'delta_p':slackvec.flatten()}, 
                                                index = np.arange(system.get('n_buses')))

        slack_distribution_df.index.name = 'bus'

        slack_gen_indices = np.array([], dtype=int)

        for i in range(len(gens.index)):
            if gens.slack[i]:
                slack_gen_indices = np.append(slack_gen_indices, gens.bus[i])

        slack_distribution_df = slack_distribution_df.filter(items = slack_gen_indices, axis = 0)
        
        #Participation factors are defined bus-wise
        #some test cases have multiple generators at a single bus
        #the line below is a workaround to avoid showing multiple busses and too much slack
        slack_distribution_df = slack_distribution_df.groupby(level=0).mean()
        print(slack_distribution_df)
        print('\n')
        
        #add the corresponding slack to each slack generator after load variation
        for i in range(len(gens.index)):
            if gens.slack[i]:
                pset = gens.pset[i]
                pset += gens.participation_factor[i] * psi_load
                gens.pset[i] = pset
        
        system.update({'generators':gens})
    else:
        print("\nTotal variation in real power load: %f pu\n" % psi_load)
    
    system.update({'loads':loads})
    
    return


def run_newton_raphson(system, enforce_q_limits = False):
    
    if enforce_q_limits == True:
        recalculate = True
        m = 0
        
        while recalculate == True:
            
            (n_buses, g, b) = process_admittance_mat(system)
        
            (vmag, delta, vmag_full, delta_full) = init_voltage_vecs(system)
        
            (p, q, p_full, q_full) = calc_power_vecs(system, vmag_full, delta_full, g, b)
        
            jacobian = calc_jacobian(system, vmag_full, delta_full, g, b, p_full, q_full)
        
            jacobian_calc = jacobian_calc_simplify(system, jacobian)
        
            (pset, qset) = calc_power_setpoints(system)
        
            (del_p, del_q) = calc_mismatch_vecs(system, p, q)
        
            #obtaining list of non-PV and non-slack busses
            pv_idx = get_pv_idx(system)
            pq_idx = np.arange(n_buses)
            non_slack_idx = np.delete(pq_idx, slack_idx(system), 0)
            pq_idx = np.delete(pq_idx, pv_idx, 0)
            pq_idx = pq_idx[pq_idx != slack_idx(system)]
            
            
            iteration_limit = system.get('iteration_limit')
            tolerance = system.get('tolerance')
            gens = system.get('generators')
        
            for i in range(1, iteration_limit + 1):
                (delta_next, vmag_next) = next_iteration(jacobian_calc, vmag, delta, del_p, del_q)
                
                if check_convergence(delta_next, vmag_next, delta, vmag, tolerance):
                    recalculate = check_pv_bus(system, n_buses, q_full)
                    
                    if recalculate:
                        print('Recalculating power flow...\n')
                        break
                    else:
                        print("Power flow converged at %d iterations (tolerance of %f).\n" % (i, tolerance))
                        #print("Mismatch vector (P injections)\n", del_p)
                        #print("Mismatch vector (Q injections)\n", del_q)
                        print("\nTable of results (power values are injections):\n")
                        
                        typelist = ['' for i in range(n_buses)]
                        typelist[system.get('slack').bus[0]] = 'SLACK'
                    
                        for i in range(len(gens.index)):
                            k = gens.bus[i]
                            typelist[k] = gens.type[i].upper()
                        
                        for i in range(n_buses):
                            if typelist[i] == '':
                                typelist[i] = 'PQ'
                        
                        delta_full[non_slack_idx] = delta_next #updating voltage angles on all busses except slack
                        vmag_full[pq_idx] = vmag_next #updating voltage magnitudes on non-slack and non-PV busses
                        
                        (p, q, p_full, q_full) = calc_power_vecs(system, vmag_full, delta_full, g, b)
                        
                        d = {'vmag_pu':vmag_full.flatten(), 'delta_deg':delta_full.flatten()*180/np.pi, 'p_pu':p_full.flatten(), 'q_pu':q_full.flatten(), 'type':typelist}
                        df = pd.DataFrame(data=d, index = np.arange(n_buses))
                        df.index.name = 'bus'
                        print(df)
                    break
                
                elif i == iteration_limit:
                    print("Power flow did not converge after %d iterations (tolerance of %f).\n" % (i, tolerance))
                    return None

                delta_full[non_slack_idx] = delta_next #updating voltage angles on all busses except slack
                vmag_full[pq_idx] = vmag_next #updating voltage magnitudes on non-slack and non-PV busses
                
                delta = np.copy(delta_next)
                vmag = np.copy(vmag_next)
                
                (p, q, p_full, q_full) = calc_power_vecs(system, vmag_full, delta_full, g, b)
        
                jacobian = calc_jacobian(system, vmag_full, delta_full, g, b, p_full, q_full)
        
                jacobian_calc = jacobian_calc_simplify(system, jacobian)
        
                (del_p, del_q) = calc_mismatch_vecs(system, p, q)
                
                y = np.row_stack((del_p, del_q))
                
            #Tracking how many times the while-loop has run
            m += 1
            if m > 40:
                print('\nError - endless loop. Calculation terminated.\n')
                break
        
    else:
        (n_buses, g, b) = process_admittance_mat(system)
    
        (vmag, delta, vmag_full, delta_full) = init_voltage_vecs(system)
    
        (p, q, p_full, q_full) = calc_power_vecs(system, vmag_full, delta_full, g, b)
    
        jacobian = calc_jacobian(system, vmag_full, delta_full, g, b, p_full, q_full)
    
        jacobian_calc = jacobian_calc_simplify(system, jacobian)
    
        (pset, qset) = calc_power_setpoints(system)
    
        (del_p, del_q) = calc_mismatch_vecs(system, p, q)
    
        #obtaining list of non-PV and non-slack busses
        pv_idx = get_pv_idx(system)
        pq_idx = np.arange(n_buses)
        non_slack_idx = np.delete(pq_idx, slack_idx(system), 0)
        pq_idx = np.delete(pq_idx, pv_idx, 0)
        pq_idx = pq_idx[pq_idx != slack_idx(system)]
        
        iteration_limit = system.get('iteration_limit')
        tolerance = system.get('tolerance')
        gens = system.get('generators')
    
        for i in range(1, iteration_limit + 1):
            (delta_next, vmag_next) = next_iteration(jacobian_calc, vmag, delta, del_p, del_q)
            
            if check_convergence(delta_next, vmag_next, delta, vmag, tolerance):
                
                print("Power flow converged at %d iterations (tolerance of %f).\n" % (i, tolerance))
                #print("Mismatch vector (P injections)\n", del_p)
                #print("Mismatch vector (Q injections)\n", del_q)
                print("\nTable of results (power values are injections):\n")
                typelist = ['' for i in range(n_buses)]
                typelist[system.get('slack').bus[0]] = 'SLACK'
            
                for i in range(len(gens.index)):
                    k = gens.bus[i]
                    typelist[k] = gens.type[i].upper()
                
                for i in range(n_buses):
                    if typelist[i] == '':
                        typelist[i] = 'PQ'
                
                delta_full[non_slack_idx] = delta_next #updating voltage angles on all busses except slack
                vmag_full[pq_idx] = vmag_next #updating voltage magnitudes on non-slack and non-PV busses
                
                (p, q, p_full, q_full) = calc_power_vecs(system, vmag_full, delta_full, g, b)
                
                d = {'vmag_pu':vmag_full.flatten(), 'delta_deg':delta_full.flatten()*180/np.pi, 'p_pu':p_full.flatten(), 'q_pu':q_full.flatten(), 'type':typelist}
                df = pd.DataFrame(data=d, index = np.arange(n_buses))
                df.index.name = 'bus'
                print(df)
                break
            
            elif i == iteration_limit:
                print("Power flow did not converge after %d iterations (tolerance of %f).\n" % (i, tolerance))
                return None
            
            delta_full[non_slack_idx] = delta_next #updating voltage angles on all busses except slack
            vmag_full[pq_idx] = vmag_next #updating voltage magnitudes on non-slack and non-PV busses
            
            delta = np.copy(delta_next)
            vmag = np.copy(vmag_next)
            
            (p, q, p_full, q_full) = calc_power_vecs(system, vmag_full, delta_full, g, b)
    
            jacobian = calc_jacobian(system, vmag_full, delta_full, g, b, p_full, q_full)
    
            jacobian_calc = jacobian_calc_simplify(system, jacobian)
    
            (del_p, del_q) = calc_mismatch_vecs(system, p, q)
            
            y = np.row_stack((del_p, del_q))   
    
    #Saving and exporting power flow results as dictionary
    vmag_res = df['vmag_pu']
    vmag_res = pd.Series.to_numpy(vmag_res)

    delta_res = df['delta_deg']
    delta_res = pd.Series.to_numpy(delta_res) * np.pi / 180
    
    p_res = df['p_pu']
    p_res = pd.Series.to_numpy(p_res)
    
    q_res = df['q_pu']
    q_res = pd.Series.to_numpy(q_res) 

    p_loss = calc_system_losses(system, vmag_res, delta_res)    

    line_flows = calc_line_flows(system, vmag_res, delta_res)
    
    trafo_flows = calc_transformer_loadings(system, vmag_res, delta_res)
    
    results = {'bus_results':df, 'line_flows':line_flows, 'transformer_flows':trafo_flows, 
               'total_losses_mw':p_loss*system.get('s_base'), 'mismatches':y}
    
    print("\nWarnings:\n")
    check_q_limits(system, q_res)
    check_bus_voltage(system, vmag_res)
    check_line_trafo_loading(system, results)
    
    return results   

def run_newton_raphson_distributed(system, enforce_q_limits = False):
    
    if enforce_q_limits == True:
        recalculate = True
        m = 0
        
        while recalculate == True:
            
            (n_buses, g, b) = process_admittance_mat(system)

            (vmag, delta, vmag_full, delta_full) = init_voltage_vecs(system)

            k_g = 0.0

            (p, q, p_full, q_full) = calc_power_vecs(system, vmag_full, delta_full, g, b)

            (pset, qset) = calc_power_setpoints(system)
            
            (del_p, del_q) = calc_mismatch_vecs(system, p, q)

            jacobian = calc_jacobian(system, vmag_full, delta_full, g, b, p_full, q_full)

            jacobian_calc = jacobian_calc_simplify(system, jacobian)


            pv_idx = get_pv_idx(system)
            pq_idx = np.arange(n_buses)
            non_ref_idx = np.delete(pq_idx, system.get('reference_bus'), 0)
            pq_idx = np.delete(pq_idx, pv_idx, 0)

            iteration_limit = system.get('iteration_limit')
            tolerance = system.get('tolerance')
            gens = system.get('generators')
        
            for i in range(1, iteration_limit + 1):
                (delta_next, vmag_next, k_g_next) = dist_next_iteration(jacobian_calc, vmag, delta, k_g, 
                                                                           del_p, del_q)
                
                if dist_check_convergence(delta_next, vmag_next, delta, vmag, k_g_next, k_g, tolerance):
                    recalculate = check_pv_bus(system, n_buses, q_full)
                    
                    if recalculate:
                        print('Recalculating power flow...\n')
                        break
                    else:
                        print("Power flow converged at %d iterations (tolerance of %f).\n" % (i, tolerance))
                        #print("Mismatch vector (P injections)\n", del_p)
                        #print("Mismatch vector (Q injections)\n", del_q)
                        print("\nTable of results (power values are injections):\n")
                        typelist = ['' for i in range(n_buses)]
                    
                        for i in range(len(gens.index)):
                            k = gens.bus[i]
                            typelist[k] = gens.type[i].upper()
                        
                        for i in range(n_buses):
                            if typelist[i] == '':
                                typelist[i] = 'PQ'
                        
                        delta_full[non_ref_idx] = delta_next #updating voltage angles on all busses except reference
                        vmag_full[pq_idx] = vmag_next #updating voltage magnitudes on non-PV busses
                        k_g = k_g_next
                        
                        (p, q, p_full, q_full) = calc_power_vecs(system, vmag_full, delta_full, g, b)
                        
                        d = {'vmag_pu':vmag_full.flatten(), 'delta_deg':delta_full.flatten()*180/np.pi, 'p_pu':p_full.flatten(), 'q_pu':q_full.flatten(), 'type':typelist}
                        df = pd.DataFrame(data=d, index = np.arange(n_buses))
                        df.index.name = 'bus'
                        print(df)
                        break
                
                elif i == iteration_limit:
                    print("Power flow did not converge after %d iterations (tolerance of %f).\n" % (i, tolerance))
                    return None
                    
                delta_full[non_ref_idx] = delta_next #updating voltage angles on all busses except reference
                vmag_full[pq_idx] = vmag_next #updating voltage magnitudes on non-PV busses
                
                delta = np.copy(delta_next)
                vmag = np.copy(vmag_next)
                k_g = k_g_next
                
                (p, q, p_full, q_full) = calc_power_vecs(system, vmag_full, delta_full, g, b)

                jacobian = calc_jacobian(system, vmag_full, delta_full, g, b, p_full, q_full)

                jacobian_calc = jacobian_calc_simplify(system, jacobian)
                
                del_p = pset - (p + slack_distribution(system, k_g))
                del_q = qset - q


            
                
            #Tracking how many times the while-loop has run
            m += 1
            if m > 40:
                print('\nError - endless loop. Calculation terminated.\n')
                break
        
    else:
        
        (n_buses, g, b) = process_admittance_mat(system)

        (vmag, delta, vmag_full, delta_full) = init_voltage_vecs(system)

        k_g = 0.0

        (p, q, p_full, q_full) = calc_power_vecs(system, vmag_full, delta_full, g, b)

        (pset, qset) = calc_power_setpoints(system)

        (del_p, del_q) = calc_mismatch_vecs(system, p, q)

        jacobian = calc_jacobian(system, vmag_full, delta_full, g, b, p_full, q_full)

        jacobian_calc = jacobian_calc_simplify(system, jacobian)


        pv_idx = get_pv_idx(system)
        pq_idx = np.arange(n_buses)
        non_ref_idx = np.delete(pq_idx, system.get('reference_bus'), 0)
        pq_idx = np.delete(pq_idx, pv_idx, 0)

        iteration_limit = system.get('iteration_limit')
        tolerance = system.get('tolerance')
        gens = system.get('generators')
    
        for i in range(1, iteration_limit + 1):
            (delta_next, vmag_next, k_g_next) = dist_next_iteration(jacobian_calc, vmag, delta, k_g, 
                                                                       del_p, del_q)
            
            if dist_check_convergence(delta_next, vmag_next, delta, vmag, k_g_next, k_g, tolerance):
                print("Power flow converged at %d iterations (tolerance of %f).\n" % (i, tolerance))
                #print("Mismatch vector (P injections)\n", del_p)
                #print("Mismatch vector (Q injections)\n", del_q)
                print("\nTable of results (power values are injections):\n")
                typelist = ['' for i in range(n_buses)]
            
                for i in range(len(gens.index)):
                    k = gens.bus[i]
                    typelist[k] = gens.type[i].upper()
                
                for i in range(n_buses):
                    if typelist[i] == '':
                        typelist[i] = 'PQ'
                
                delta_full[non_ref_idx] = delta_next #updating voltage angles on all busses except reference
                vmag_full[pq_idx] = vmag_next #updating voltage magnitudes on non-PV busses
                k_g = k_g_next
                
                (p, q, p_full, q_full) = calc_power_vecs(system, vmag_full, delta_full, g, b)
                
                d = {'vmag_pu':vmag_full.flatten(), 'delta_deg':delta_full.flatten()*180/np.pi, 'p_pu':p_full.flatten(), 'q_pu':q_full.flatten(), 'type':typelist}
                df = pd.DataFrame(data=d, index = np.arange(n_buses))
                df.index.name = 'bus'
                print(df)
                break
            
            elif i == iteration_limit:
                print("Power flow did not converge after %d iterations (tolerance of %f).\n" % (i, tolerance))
                return None
                
            delta_full[non_ref_idx] = delta_next #updating voltage angles on all busses except reference
            vmag_full[pq_idx] = vmag_next #updating voltage magnitudes on non-PV busses
            
            delta = np.copy(delta_next)
            vmag = np.copy(vmag_next)
            k_g = k_g_next
            
            (p, q, p_full, q_full) = calc_power_vecs(system, vmag_full, delta_full, g, b)

            jacobian = calc_jacobian(system, vmag_full, delta_full, g, b, p_full, q_full)

            jacobian_calc = jacobian_calc_simplify(system, jacobian)
            
            del_p = pset - (p + slack_distribution(system, k_g))
            del_q = qset - q  
    
    #Saving and exporting power flow results as dictionary
    vmag_res = df['vmag_pu']
    vmag_res = pd.Series.to_numpy(vmag_res)

    delta_res = df['delta_deg']
    delta_res = pd.Series.to_numpy(delta_res) * np.pi / 180
    
    p_res = df['p_pu']
    p_res = pd.Series.to_numpy(p_res)
    
    q_res = df['q_pu']
    q_res = pd.Series.to_numpy(q_res) 

    p_loss = calc_system_losses(system, vmag_res, delta_res)    

    line_flows = calc_line_flows(system, vmag_res, delta_res)
    
    trafo_flows = calc_transformer_loadings(system, vmag_res, delta_res)
    
    slack_distribution_df = pd.DataFrame(data={'p_pu':(-1)*slack_distribution(system, k_g).flatten()}, 
                                            index = np.arange(n_buses))

    slack_distribution_df.index.name = 'bus'

    slack_gen_indices = np.array([], dtype=int)

    for i in range(len(gens.index)):
        if gens.slack[i]:
            slack_gen_indices = np.append(slack_gen_indices, gens.bus[i])

    slack_distribution_df = slack_distribution_df.filter(items = slack_gen_indices, axis = 0)
    
    #Participation factors are defined bus-wise
    #some test cases have multiple generators at a single bus
    #the line below is a workaround to avoid showing multiple busses and too much slack
    slack_distribution_df = slack_distribution_df.groupby(level=0).mean()

    print("\nSlack (losses) distribution across slack generators:\n")
    print(slack_distribution_df)
    
    results = {'bus_results':df, 'line_flows':line_flows, 'total_losses_pu':p_loss, 'transformer_flows':trafo_flows,
               'mismatches':calc_mismatch_vecs(system, p, q), 'slack_distribution':slack_distribution_df}

    
    print("\nWarnings:\n")
    check_p_limits(system, p_res)
    check_q_limits(system, q_res)
    check_bus_voltage(system, vmag_res)
    check_line_trafo_loading(system, results)
    
    return results 


def run_power_flow(system, enforce_q_limits = False, distributed_slack = False):
    #Master function
    
    if distributed_slack:
        results = run_newton_raphson_distributed(system, enforce_q_limits)
    else:
        results = run_newton_raphson(system, enforce_q_limits)
    
    return results


def plot_results(system, results):
    
    #Bus voltages
    plt.figure()
    plt.scatter(results.get('bus_results').index, results.get('bus_results')['vmag_pu'], marker="D", 
                color='mediumblue',s=25)
    plt.scatter(system.get('buses').index, system.get('buses')['max_vm_pu'], marker="_", color='tab:red',s=30)
    plt.scatter(system.get('buses').index, system.get('buses')['min_vm_pu'], marker="_", color='tab:red',s=30)
    plt.title('Results - Bus Voltage Magnitudes')
    plt.ylabel('Magnitude [p.u.]')
    plt.xlabel('Bus')
    plt.xticks(range(0, system.get('n_buses'), 2))
    plt.grid(linestyle='--', linewidth=0.5, alpha=0.65)
    plt.margins(x=0.025)
    plt.show()
    
    #Line loadings
    plt.figure()
    plt.scatter(results.get('line_flows').index, results.get('line_flows')['loading_percent'], marker="D", 
                color='teal',s=25)
    plt.scatter(results.get('line_flows').index, np.ones(len(results.get('line_flows').index))*100, marker="_", color='tab:red',s=30)
    if  max(results.get('line_flows')['loading_percent']) > 100:
        plt.ylim(0,max(results.get('line_flows')['loading_percent']) + 5)
    else:
        plt.ylim(0,110)
    plt.title('Results - Line Loading')
    plt.ylabel('Percentage')
    plt.xlabel('Line')
    plt.xticks(range(0, len(results.get('line_flows').index), 2))
    plt.grid(linestyle='--', linewidth=0.5, alpha=0.65)
    plt.margins(x=0.025)
    plt.show()

    #Transformer loadings
    plt.figure()
    plt.scatter(results.get('transformer_flows').index, results.get('transformer_flows')['loading_percent'], marker="D", 
                color='darkgreen',s=25)
    plt.scatter(results.get('transformer_flows').index, np.ones(len(results.get('transformer_flows').index))*100, marker="_", color='tab:red',s=60)
    if  max(results.get('transformer_flows')['loading_percent']) > 100:
        plt.ylim(0,max(results.get('transformer_flows')['loading_percent']) + 5)
    else:
        plt.ylim(0,110)
    plt.title('Results - Transformer Loading')
    plt.ylabel('Percentage')
    plt.xlabel('Transformer')
    plt.xticks(range(0, len(results.get('transformer_flows').index), 1))
    plt.grid(linestyle='--', linewidth=0.5, alpha=0.65)
    plt.margins(x=0.025)
    plt.show()
    
    return

def plot_result_comparison(system, results1, results2):
    
    #Plots differences for bus voltages, line loadings etc. between two result dictionaries
    
    return

#=============================================================================

#Unused functions

# def calc_power_from_to(system, vmag, delta, from_idx, to_idx):
#     #Note - vmag and delta should be the full vectors
#     (n, g, b) = process_admittance_mat(system)
#     i = from_idx
#     j = to_idx
    
#     p = vmag[i] ** 2 * g[i,j] - vmag[i] * vmag[j] * (
#         g[i,j] * np.cos(delta[i] - delta[j]) + b[i,j] * np.sin(delta[i] - delta[j]))
    
#     q = - (vmag[i] ** 2) * b[i,j] - vmag[i] * vmag[j] * (
#         g[i,j] * np.sin(delta[i] - delta[j]) - b[i,j] * np.cos(delta[i] - delta[j]))

#     s = complex(p,q)
    
#     return s, p, q


# def calc_abcd_param(system, line_idx):
#     line = system.get('lines')[line_idx]
#     omega = 2*np.pi*system.get('frequency') #support for non-50Hz systems
#     l = line.get('length') #get line length in km
    
#     z = complex(line.get('r_ohm_per_km'), line.get('x_ohm_per_km'))
#     y = complex(line.get('g_us_per_km')*1e-6, omega*line.get('c_nf_per_km')*1e-9)
#     Y = y*l
#     Z = z*l
    
#     if l <= 25: #short line 
#         a = 1
#         d = 1
#         b = Z
#         c = 0
#     elif l <= 250: #medium line - nominal parameters
#         a = 1 + (Y*Z)/2
#         d = 1 + (Y*Z)/2
#         b = Z
#         c = Y * (1 + (Y*Z)/4)
        
#     else: #long line - exact parameters
#         gamma = np.sqrt(z*y)
#         z_c = np.sqrt(z/y)
#         a = np.cosh(gamma*l)
#         d = np.cosh(gamma*l)
#         b = z_c * np.sinh(gamma*l)
#         c = (1/z_c) * np.sinh(gamma*l)
    
#     return a, b, c, d

