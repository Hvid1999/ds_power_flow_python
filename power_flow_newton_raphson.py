#This function library was written as part of a bachelor thesis at the
#Technical University of Denmark (DTU) during the Spring semester of 2022.
#The code is meant to be easily read and is built around the use of dictionaries
#and dataframes. As of now, systems are loaded directly from PandaPower, but 
#expanding to load custom systems should be relatively straightforward.

#Author: Markus Hvid Monin (s194011)

import numpy as np
import pandas as pd
import pandapower as pp
import matplotlib.pyplot as plt
import matplotlib.gridspec as gsp


# =============================================================================
# Functions for loading and tweaking PandaPower systems

def load_pandapower_case(network, enforce_q_limits=False, distributed_slack = False, 
                         slack_gens = np.array([]), participation_factors = np.array([]),
                         ref_bus_pset = 0, original_ref_bus = False):
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

    
    #Calculating transformer short circuit impedance and magnitizing admittance
    #used for building admittance matrix through custom code
    #based on https://pandapower.readthedocs.io/en/develop/elements/trafo.html 
    #"The values calculated in that way are relative to the rated values of the transformer. 
    # To transform them into the per unit system, 
    # they have to be converted to the rated values of the network."
    
    z = trafo['vk_percent']/100 * baseMVA/trafo['sn_mva']
    r = trafo['vkr_percent']/100 * baseMVA/trafo['sn_mva'] 
    x = np.sqrt(z ** 2 - r ** 2)
    z_k = (1 + 0j)*r + 1j*x 
    trafo['z_k'] = round(z_k,6)
    
    y = trafo['i0_percent']/100
    g = trafo['pfe_kw']/(trafo['sn_mva']*1e3) * baseMVA/trafo['sn_mva']
    b = np.sqrt(y ** 2 - g ** 2)
    y_m = (1 + 0j)*g + 1j*b
    trafo['y_m'] = round(y_m,6)
    
    #Reformatting transformers dataframe
    trafo = trafo.rename(columns={'sn_mva':'s_rated'})
    trafo = trafo[['in_service', 'lv_bus', 'hv_bus', 'parallel', 's_rated', 'tap_pos', 'tap_min', 'tap_max',
                   'tap_side', 'tap_step_percent','z_k', 'y_m','vk_percent','vkr_percent','i0_percent','pfe_kw']]

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
        #Note: If a single slack bus is entered, the single slack power flow is obtained.
        #Changing slack bus to PV-bus
        #The setpoint original slack bus generator is difference between total load and total generation
        load_sum = 0
        gen_sum = 0

        for i in range(len(load.index)):
            load_sum += load.p[i]

        for i in range(len(gens.index)):
            gen_sum += gens.pset[i]

        slack_pset = load_sum - gen_sum
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
        
        
        if (np.size(slack_gens) == 1) and (slack.bus[0] != gens.bus[slack_gens[0]]) and not original_ref_bus:
            system.update({'reference_bus':system.get('generators').bus[slack_gens[0]]})
        else:
            #If a single slack gen is specified, it is saved as the angle reference bus,
            #otherwise the original single slack bus is kept as angle reference
            system.update({'reference_bus':slack.bus[0]})
        
        del system['slack'] #removing separate slack bus description
        
        if ref_bus_pset != 0:
            set_reference_bus_power(system, ref_bus_pset)
        
        load_participation_factors(system, p_factors=participation_factors) #loading either equal p-factors or custom ones

    return (system, pandapower_results)


def load_participation_factors(system, p_factors = np.array([])):
    #accepts an array of participation factors ordered by increasing generator bus indices
    #if no array is entered, slack is distributed evenly among generators participating in slack
    gens = system.get('generators')
    
    slack_gens = gens[gens.slack & gens.in_service] 
    
    num_slack = len(slack_gens.index)
    
    if np.any(p_factors < 0):
        print('Error loading participation factors - all values must be positive.' % sum(p_factors))
        print('Set to equal factors (standard case).\n')
        participation_factors = np.ones(num_slack)
        participation_factors = participation_factors / num_slack
    elif np.size(p_factors) == 0: #standard case for no input
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


def set_reference_bus_power(system, pset):
    if not system.get('distributed_slack'):
        print("Error - single slack bus system. Cannot slack generator power.\n")
        return
    else:
        gens = system.get('generators')
        idx = gens[gens.bus == system.get('reference_bus')].index[0]
        gens.pset[idx] = pset
        system.update({'generators':gens.copy()})
        return
        

def new_england_39_new_voltages(network):
    #https://pdfcoffee.com/39-bus-new-england-system-pdf-free.html
    #The PandaPower nominal bus voltages are all set to 345 kV
    #the change makes little difference for power flow, but it may be more
    #accurate.
    
    #Fixing bus base voltages
    network.bus['vn_kv'][11] = 138
    network.bus['vn_kv'][19] = 230
    network.bus['vn_kv'][29] = 16.5
    network.bus['vn_kv'][30] = 16.5
    network.bus['vn_kv'][31] = 16.5
    network.bus['vn_kv'][32] = 16.5
    network.bus['vn_kv'][33] = 16.5
    network.bus['vn_kv'][34] = 16.5
    network.bus['vn_kv'][35] = 16.5
    network.bus['vn_kv'][36] = 16.5
    network.bus['vn_kv'][37] = 16.5

    #Updating transformer voltages
    network.trafo['vn_lv_kv'][0] = 16.5
    network.trafo['vn_lv_kv'][1] = 16.5
    network.trafo['vn_lv_kv'][2] = 16.5
    network.trafo['vn_lv_kv'][3] = 138
    network.trafo['vn_lv_kv'][4] = 138
    network.trafo['vn_lv_kv'][5] = 230
    network.trafo['vn_lv_kv'][6] = 16.5
    network.trafo['vn_lv_kv'][7] = 16.5
    network.trafo['vn_hv_kv'][7] = 230
    network.trafo['vn_lv_kv'][8] = 16.5
    network.trafo['vn_lv_kv'][9] = 16.5
    network.trafo['vn_lv_kv'][10] = 16.5

    #Fixing erroneously defined transformer sides
    network.trafo['lv_bus'][3] = 11
    network.trafo['hv_bus'][3] = 10
    network.trafo['lv_bus'][4] = 11
    network.trafo['hv_bus'][4] = 12
    
    return network


def new_england_case_line_fix(system):
    #Change transformer erroneously defined as transmission line i PandaPower
    lines = system.get('lines')
    #saving short circuit impedance value
    z_k = np.round(complex(lines['r_ohm_per_km'][29],
                  lines['x_ohm_per_km'][29]) / (345000**2 / 100000000),6)
    lines = lines.drop(labels=29, axis=0)
    lines = lines.reset_index(drop=True)
    system.update({'lines':lines})

    trafo = system.get('transformers')
    swap_trafo = trafo[trafo.index == 2].reset_index(drop=True) #copy other trafo with same rating - see matpower desc
    swap_trafo['hv_bus'][0] = 22
    swap_trafo['lv_bus'][0] = 35
    swap_trafo['z_k'] = z_k
    swap_trafo['in_service'] = True
    trafo = trafo.append(swap_trafo)
    trafo = trafo.sort_values(by=['hv_bus'])
    trafo = trafo.reset_index(drop=True)
    system.update({'transformers':trafo})
    
    return


def toggle_element(system, element, index):
    #Toggles the element status at input index in the relevant dataframe
    if element == 'generator':
        gens = system.get('generators')
        gens.in_service[index] = not gens.in_service[index]
    elif element == 'transformer':
        trafo = system.get('transformer')
        trafo.in_service[index] = not trafo.in_service[index]
    elif element == 'line':
        line = system.get('lines')
        line.in_service[index] = not line.in_service[index]
    elif element == 'load':
        load = system.get('loads')
        load.in_service[index] = not load.in_service[index]
    elif element == 'bus':
        bus = system.get('buses')
        bus.in_service[index] = not bus.in_service[index]
    else:
        print('Invalid element. Check input.')
    
    
    #Update admittance matrix according to change
    
    return


# =============================================================================
# Functions for extracting system information


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
        if (gens.type[i] == 'pv') and gens.in_service[i]:
            pv_idx = np.append(pv_idx, gens.bus[i])
    
    return pv_idx

def slack_idx(system):
    if system.get('distributed_slack'):
        return system.get('reference_bus')
    else:
        return system.get('slack').bus[0]


def build_admittance_matrix(system):
    #line model: pi
    #trafo model: series inductance / pi (simple - no tap changing)
    
    #1 - calculate per unitized values according to base impedances
    #2 - build off-diagonals
    #3 - build diagonals
    
    #UNFINISHED
    
    n_buses = len(system.get('buses').index)
    line = system.get('lines')
    trafo = system.get('transformers')
    bus = system.get('buses')
    s_base = system.get('s_base')
    
    ybus = np.zeros((n_buses, n_buses), dtype = complex)
    
    for i in range(len(line.index)):
        if line['in_service'][i]:
            fr = line['from'][i]
            to = line['to'][i]
            z_base = ((bus.vn_kv[fr] * 1e3) ** 2) / (s_base * 1e6)
            
            z_line_pu = complex(line['r_ohm_per_km'][i], 
                                line['x_ohm_per_km'][i]) * line.length[i] / z_base
            y_line_pu = complex(line['g_us_per_km'][i], 
                                2*np.pi*system.get('frequency')*line['c_nf_per_km'][i]*1e-9
                                ) * line.length[i] * z_base
            
            #diagonals
            ybus[fr, fr] += 1 / z_line_pu + 0.5 * y_line_pu
            ybus[to, to] += 1 / z_line_pu + 0.5 * y_line_pu
            
            #off-diagonals
            ybus[fr, to] -= 1 / z_line_pu
            ybus[to, fr] -= 1 / z_line_pu
        
        
    for i in range(len(trafo.index)):
        #Note: values of z_k and y_m for the transformers are per unitized 
        #with respect to the low voltage side
        #Per unit impedance remains unchanged when referred from one side to the other
        
        if trafo['in_service'][i]:
            lv = trafo['lv_bus'][i]
            hv = trafo['hv_bus'][i]
            
            # #per-unitization of transformer values
            # #based on PandaPower documentation
            # z_base_lv = bus.vn_kv[lv] ** 2 / s_base
            # z_ref_trafo = bus.vn_kv[lv] ** 2 * s_base / trafo.s_rated[i]
            
            # # z_pu = trafo.z_k[i] * z_ref_trafo/z_base_lv
            # # y_pu = trafo.y_m[i] * z_base_lv/z_ref_trafo
            z_pu = trafo.z_k[i]
            y_pu = trafo.y_m[i]
            
            #diagonals
            ybus[lv, lv] += 1 / z_pu + 0.5 * y_pu
            ybus[hv, hv] += 1 / z_pu + 0.5 * y_pu
            
            #off-diagonals
            ybus[lv, hv] -= 1 / z_pu
            ybus[hv, lv] -= 1 / z_pu
        
    
    #Further work: code handlings shunts...
    #not relevant for new england 39 bus system case
    
    return ybus


# =============================================================================
# Functions for Newton-Raphson power flow calculations


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
        if (gens.type[i] == 'pv') and gens.in_service[i]:
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
    
    for i in range(len(gens.index)):
        if (gens.type[i] == 'pv') and (gens.in_service[i]): #only considering in service PV-busses
            k = gens.bus[i]
            
            #Checking for static generators on the PV gen bus (not relevant for New England system)
            sgen = gens[(gens.bus == k) & (gens.type == 'pq')].reset_index(drop=True)
            if len(sgen.index) != 0:
                q_sgen = 0
                for i in range(len(sgen.index)):
                    q_sgen += sgen.qset[i]
                q_gen = q_full[k] - q_loads[k] - q_sgen
            else:
                q_gen = q_full[k] - q_loads[k]
            
            if q_gen < gens.qmin[i]:
                qset = gens.qmin[i]
                gens.qset[i] = qset
                gens.type[i] = 'pq'
                limit_violation = True
                break
            elif q_gen > gens.qmax[i]:
                qset = gens.qmax[i]
                gens.qset[i] = qset
                gens.type[i] = 'pq'
                limit_violation = True
                break
            system.update({'generators':gens})
    
    if limit_violation == True:
        print('Generator reactive power limit violated at bus %d (%f pu).\nType set to PQ with generator reactive power setpoint of %.2f pu.\n' % (k, q_gen, qset))
    
    return limit_violation


def run_power_flow(system, enforce_q_limits = False, distributed_slack = False):
    #Master function
    
    if distributed_slack:
        print("-------------------------------------------------------------")
        print("Calculating power flow (distributed slack bus)...\n")
        results = run_newton_raphson_distributed(system, enforce_q_limits)
    else:
        print("-------------------------------------------------------------")
        print("Calculating power flow (single slack bus)...\n")
        results = run_newton_raphson(system, enforce_q_limits)
    
    return results

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
               'total_losses_pu':p_loss, 'mismatches':y}
    
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

    print("\nSlack (%f p.u.) distribution across slack generators:\n" % (-1*k_g))
    print(slack_distribution_df)
    
    results = {'bus_results':df, 'line_flows':line_flows, 'total_losses_pu':p_loss, 'transformer_flows':trafo_flows,
               'mismatches':calc_mismatch_vecs(system, p, q), 'slack_distribution':slack_distribution_df}

    
    print("\nWarnings:\n")
    check_p_limits(system, p_res)
    check_q_limits(system, q_res)
    check_bus_voltage(system, vmag_res)
    check_line_trafo_loading(system, results)
    
    return results 



# =============================================================================
# Functions for evaluating power flow results

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
        if gens.type[i] == 'pv':
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


def get_phasor(vmag, delta_rad, bus):
    return complex(vmag[bus]*np.cos(delta_rad[bus]),vmag[bus]*np.sin(delta_rad[bus]))


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
    fr = np.zeros(n_lines, dtype = int)
    to = np.zeros(n_lines, dtype = int)
    loading_percent = np.zeros(n_lines)
    
    fr = lines['from'].to_numpy()
    to = lines['to'].to_numpy()

    for i in range(n_lines):
        if lines.in_service[i]:
            l = lines.length[i]
            parallel = lines.parallel[i] #number of lines in parallel
            
            #relevant base values for per unit calculations
            v_base = system.get('buses').vn_kv[fr[i]]
            z_base = (v_base ** 2)  / (s_base) #voltage in kV and power in MVA
            # I_base = S_base_3ph / sqrt(3) * V_base_LL
            i_base_ka = s_base * 1e3 / (np.sqrt(3) * v_base * 1e3) #base current in kA (power base multiplied by 1e3 instead of 1e6)
            
            
            y_shunt = complex(lines['g_us_per_km'][i] * 1e-6, 
                              2 * np.pi * freq * lines['c_nf_per_km'][i]*1e-9) * l * parallel
            y_shunt_pu =  y_shunt * z_base # Y = 1/Z, so Y_pu = 1/Z_pu = Y * Z_base
            
            z_line = complex(lines['r_ohm_per_km'][i], lines['x_ohm_per_km'][i]) * l / parallel
            z_line_pu = z_line / z_base
            
            #loading voltage magnitude and phase angle as phasor
            v_1 = get_phasor(vmag, delta, fr[i])
            v_2 = get_phasor(vmag, delta, to[i])
            
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
        
    d = {'from':fr,'to':to,'loading_percent':loading_percent, 'i_ka':i_ka, 'p_ft_pu':p_ft_pu, 'p_tf_pu':p_tf_pu, 
         'p_loss_pu':p_loss, 'q_ft_pu':q_ft_pu, 'q_tf_pu':q_tf_pu, 'i_ft_pu':np.abs(i_ft_pu), 
         'i_tf_pu':np.abs(i_tf_pu), 's_ft_pu':np.abs(s_ft_pu), 
         's_tf_pu':np.abs(s_tf_pu)}
    df = pd.DataFrame(data=d, index = np.arange(n_lines))
    df.index.name = 'line'
    
    return df


def calc_transformer_loadings(system, vmag, delta):
    #Note: Simplified representation of transformer as a series impedance between busses
    #(typical per-unit representation)

    
    trafo = system.get('transformers')
    bus = system.get('buses')
    ybus = system.get('admmat')
    s_base = system.get('s_base')
    n_trafo = len(trafo.index)
    
    #initializing empty arrays for storing data
    i_lv_pu = np.zeros(n_trafo, dtype = complex)
    i_hv_pu = np.zeros(n_trafo, dtype = complex)
    i_lv_ka = np.zeros(n_trafo, dtype = complex)
    i_hv_ka = np.zeros(n_trafo, dtype = complex)

    s_lv_pu = np.zeros(n_trafo, dtype = complex)
    s_hv_pu = np.zeros(n_trafo, dtype = complex)
    
    loading_percent = np.zeros(n_trafo)
    
    lv = trafo['lv_bus'].to_numpy()
    hv = trafo['hv_bus'].to_numpy()
    
    for i in range(n_trafo):
        v_lv = get_phasor(vmag, delta, lv[i])
        v_hv = get_phasor(vmag, delta, hv[i])
        x_t = 1 / (-1 * ybus[lv[i], hv[i]]) #loading the series impedance from the admittance matrix
        
        i_lv_pu[i] = (v_lv - v_hv) / x_t
        i_hv_pu[i] = (v_hv - v_lv) / x_t
        i_lv_ka[i] = i_lv_pu[i] * s_base / (np.sqrt(3) * bus.vn_kv[lv[i]])
        i_hv_ka[i] = i_hv_pu[i] * s_base / (np.sqrt(3) * bus.vn_kv[hv[i]])
        
        s_lv_pu[i] = v_lv * np.conj(i_lv_pu[i])
        s_hv_pu[i] = v_hv * np.conj(i_hv_pu[i])
        
        s_mva = abs(max(s_lv_pu[i], s_hv_pu[i]) * s_base)
        
        loading_percent[i] = (s_mva / trafo['s_rated'][i]) * 100
    
    d = {'lv':lv,'hv':hv,'loading_percent':loading_percent, 'p_lv_pu':np.real(s_lv_pu), 
         'p_hv_pu':np.real(s_hv_pu), 'q_lv_pu':np.imag(s_lv_pu), 
         'q_hv_pu':np.imag(s_hv_pu), 'i_lv_ka':np.abs(i_lv_ka), 
         'i_hv_ka':np.abs(i_hv_ka), 's_lv_pu':np.abs(s_lv_pu), 
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


# =============================================================================
# Functions for plotting results: Bus voltages, line/trafo loadings


def plot_results(system, results, angle=False ,name='', save_directory=''):
    #Note: need small changes if the system does not have transformers 
    #or if the system has different bus voltage limits for each bus
    
    
    if angle:
        gs = gsp.GridSpec(3, 2)
    else:
        gs = gsp.GridSpec(2, 2)
    fig = plt.figure(dpi=200)
    fig.set_figheight(11)
    fig.set_figwidth(11)
    if name != '':
        plt.title('%s\n\n' % name, fontweight='bold', fontsize=14)
        ax = plt.gca()
        ax.get_xaxis().set_visible(False)
        ax.get_yaxis().set_visible(False)
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.spines['bottom'].set_visible(False)
        ax.spines['left'].set_visible(False)
        
    ax1 = fig.add_subplot(gs[0, :]) # row 0, col 0
    ax1.scatter(results.get('bus_results').index, results.get('bus_results')['vmag_pu'], marker="D", 
                color='darkblue',s=25)
    # ax1.scatter(system.get('buses').index, system.get('buses')['max_vm_pu'], marker="_", color='tab:red',s=30)
    # ax1.scatter(system.get('buses').index, system.get('buses')['min_vm_pu'], marker="_", color='tab:red',s=30)
    ax1.axhline(y=system.get('buses')['max_vm_pu'][0], color='tab:red', linestyle='--')
    ax1.axhline(y=system.get('buses')['min_vm_pu'][0], color='tab:red', linestyle='--')
    ax1.title.set_text('Bus Voltage')
    ax1.set_ylabel('Magnitude [p.u.]')
    ax1.set_xlabel('Bus')
    ax1.set_xticks(range(0, system.get('n_buses'), 2))
    ax1.grid(linestyle='--', linewidth=0.5, alpha=0.65)
    ax1.margins(x=0.025)
    
    if angle:
        ax2 = fig.add_subplot(gs[2, 0])
    else:
        ax2 = fig.add_subplot(gs[1, 0])
    ax2.bar(results.get('line_flows').index, results.get('line_flows')['loading_percent'], 
                color='teal')
    # plt.scatter(results.get('line_flows').index, np.ones(len(results.get('line_flows').index))*100, marker="_", color='tab:red',s=30)
    ax2.axhline(y=100, color='tab:red', linestyle='--')
    if  max(results.get('line_flows')['loading_percent']) > 100:
        ax2.set_ylim(0,max(results.get('line_flows')['loading_percent']) + 5)
    else:
        ax2.set_ylim(0,110)
    ax2.title.set_text('Line Loading')
    ax2.set_ylabel('Percentage')
    ax2.set_xlabel('Line')
    ax2.set_xticks(range(0, len(results.get('line_flows').index), 2))
    ax2.grid(linestyle='--', linewidth=0.5, alpha=0.65)
    ax2.margins(x=0.025)
    
    if angle:
        ax3 = fig.add_subplot(gs[2,1])
    else:
        ax3 = fig.add_subplot(gs[1,1])
    ax3.bar(results.get('transformer_flows').index, results.get('transformer_flows')['loading_percent'], 
                color='darkgreen')
    # plt.scatter(results.get('transformer_flows').index, np.ones(len(results.get('transformer_flows').index))*100, marker="_", color='tab:red',s=60)
    ax3.axhline(y=100, color='tab:red', linestyle='--')
    if  max(results.get('transformer_flows')['loading_percent']) > 100:
        ax3.set_ylim(0,max(results.get('transformer_flows')['loading_percent']) + 5)
    else:
        ax3.set_ylim(0,110)
    ax3.title.set_text('Transformer Loading')
    ax3.set_ylabel('Percentage')
    ax3.set_xlabel('Transformer')
    ax3.set_xticks(range(0, len(results.get('transformer_flows').index), 1))
    ax3.grid(linestyle='--', linewidth=0.5, alpha=0.65)
    ax3.margins(x=0.025)
    
    if angle:
        ax4 = fig.add_subplot(gs[1,:])
        ax4.bar(results.get('bus_results').index, results.get('bus_results')['delta_deg'], 
                    color='darkslateblue')
        ax4.axhline(y=0, color='darkslategray', linestyle='-')
        ax4.set_ylabel('Phase Angle [Deg.]')
        ax4.set_xlabel('Bus')
        ax4.set_xticks(range(0, system.get('n_buses'), 2))
        ax4.grid(linestyle='--', linewidth=0.5, alpha=0.65)
        ax4.margins(x=0.025)
        ax4.set_ylim(-(max(abs(results.get('bus_results')['delta_deg']))+1), 
                     max(abs(results.get('bus_results')['delta_deg']))+1)
    
    if save_directory != '':
        fig.savefig(save_directory)
    return

def plot_result_comparison(results1, results2, angle=False, name = ''):
    #Plots differences for bus voltages, line loadings etc. between two result dictionaries
    
    #Bus voltages
    vmag1 = results1.get('bus_results')['vmag_pu'].to_numpy()
    vmag2 = results2.get('bus_results')['vmag_pu'].to_numpy()
    vmag_diff = np.zeros(np.size(vmag1))
    
    for i in range(np.size(vmag1)):
        # if vmag1[i] > vmag2[i]:
        #     vmag_diff[i] = vmag1[i] - vmag2[i]
        # elif vmag2[i] > vmag1[i]:
        #     vmag_diff[i] = vmag2[i] - vmag1[i]
        vmag_diff[i] = vmag2[i] - vmag1[i]
        
    #Line loadings
    l1 = results1.get('line_flows')['loading_percent'].to_numpy()
    l2 = results2.get('line_flows')['loading_percent'].to_numpy()
    l_diff = np.zeros(np.size(l1))
    
    for i in range(np.size(l1)):
        # if l1[i] > l2[i]:
        #     l_diff[i] = l1[i] - l2[i]
        # elif l2[i] > l1[i]:
        #     l_diff[i] = l2[i] - l1[i]
        l_diff[i] = l2[i] - l1[i]
      
    #Transformer loadings
    t1 = results1.get('transformer_flows')['loading_percent'].to_numpy()
    t2 = results2.get('transformer_flows')['loading_percent'].to_numpy()
    t_diff = np.zeros(np.size(t1))
    
    for i in range(np.size(t1)):
        # if t1[i] > t2[i]:
        #     t_diff[i] = t1[i] - t2[i]
        # elif t2[i] > t1[i]:
        #     t_diff[i] = t2[i] - t1[i]
        t_diff[i] = t2[i] - t1[i]
        
    
    if angle:
        gs = gsp.GridSpec(3, 2)
    else:
        gs = gsp.GridSpec(2, 2)
    fig = plt.figure(dpi=200)
    fig.set_figheight(11)
    fig.set_figwidth(11)
    if name != '':
        plt.title('%s\n\n' % name, fontweight='bold', fontsize=14)
        ax = plt.gca()
        ax.get_xaxis().set_visible(False)
        ax.get_yaxis().set_visible(False)
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.spines['bottom'].set_visible(False)
        ax.spines['left'].set_visible(False)
    
    ax1 = fig.add_subplot(gs[0, :]) # row 0, col 0
    ax1.bar(np.arange(0,np.size(vmag_diff)), vmag_diff, color='darkblue')
    ax1.axhline(y=0, color='darkslategray', linestyle='-')
    ax1.title.set_text('Difference in Bus Voltages')
    ax1.set_ylabel('\u0394 [p.u.]')
    ax1.set_xlabel('Bus')
    ax1.set_xticks(range(0, np.size(vmag_diff), 2))
    ax1.grid(linestyle='--', linewidth=0.5, alpha=0.65)
    ax1.margins(x=0.025)
    ax1.set_ylim(-max(abs(vmag_diff)), max(abs(vmag_diff)))
    
    if angle:
        ax2 = fig.add_subplot(gs[2, 0])
    else:
        ax2 = fig.add_subplot(gs[1, 0])
    ax2.bar(np.arange(0,np.size(l_diff)), l_diff, color='teal')
    ax2.axhline(y=0, color='darkslategray', linestyle='-')
    ax2.title.set_text('Difference in Line Loading Percentage')
    ax2.set_ylabel('\u0394 [%]')
    ax2.set_xlabel('Line')
    ax2.set_xticks(range(0, np.size(l_diff), 2))
    ax2.grid(linestyle='--', linewidth=0.5, alpha=0.65)
    ax2.margins(x=0.025)
    ax2.set_ylim(-max(abs(l_diff)), max(abs(l_diff)))
    
    if angle:
        ax3 = fig.add_subplot(gs[2,1])
    else:
        ax3 = fig.add_subplot(gs[1,1])
    ax3.bar(np.arange(0,np.size(t_diff)), t_diff, color='darkgreen')
    ax3.axhline(y=0, color='darkslategray', linestyle='-')
    ax3.title.set_text('Difference in Transformer Loading Percentage')
    ax3.set_ylabel('\u0394 [%]')
    ax3.set_xlabel('Transformer')
    ax3.set_xticks(range(0, np.size(t_diff), 2))
    ax3.grid(linestyle='--', linewidth=0.5, alpha=0.65)
    ax3.margins(x=0.025)
    ax3.set_ylim(-max(abs(t_diff)), max(abs(t_diff)))
    
    if angle:
        #Bus voltage angles
        delta1 = results1.get('bus_results')['delta_deg'].to_numpy()
        delta2 = results2.get('bus_results')['delta_deg'].to_numpy()
        delta_diff = np.zeros(np.size(delta1))
        
        for i in range(np.size(delta1)):
            delta_diff[i] = delta2[i] - delta1[i]
    
        ax4 = fig.add_subplot(gs[1, :]) # row 0, col 0
        ax4.bar(np.arange(0,np.size(delta_diff)), delta_diff, color='darkslateblue')
        ax4.axhline(y=0, color='darkslategray', linestyle='-')
        ax4.set_ylabel('\u0394 [Deg.]')
        ax4.set_xlabel('Bus')
        ax4.set_xticks(range(0, np.size(delta_diff), 2))
        ax4.grid(linestyle='--', linewidth=0.5, alpha=0.65)
        ax4.margins(x=0.025)
        ax4.set_ylim(-max(abs(delta_diff)), max(abs(delta_diff)))
        
    
    return


# =============================================================================
# Functions used for testing

def load_variation(system, load_indices=np.array([]), scalings=np.array([]), const_pf = True, all_loads = False, scale = 1):
    #accepts an array of load indices to scale and an array of the 
    #corresponding scaling factors
    #Scale of 1.0 = 100%, 1.10 = 110% (+10%) etc.
    
    loads = system.get('loads')
    psi_load = 0
    
    if all_loads:
        for i in range(len(loads.index)):
            p_old = loads.p[i]
            p_new = p_old * scale
            loads.p[i] = p_new
            psi_load += p_new - p_old
            
            if const_pf:
                q_old = loads.q[i]
                q_new = q_old * scale
                loads.q[i] = q_new
            
            loads.pf[i] = loads.p[i]/(np.sqrt(loads.p[i]**2 + loads.q[i]**2)) #update pf
            
        print("\nUniform load change of %f %%." % ((scale - 1)*100))
              
    else:  
        j = 0   
        for i in load_indices:
            p_old = loads.p[i]
            p_new = p_old * scalings[j]
            loads.p[i] = p_new
            psi_load += p_new - p_old
            
            if const_pf:
                q_old = loads.q[i]
                q_new = q_old * scalings[j]
                loads.q[i] = q_new
                # print("\nLoad at bus %i changed from %f to %f (real power)\nAnd %f to %f (reactive power)." % (loads.bus[i], p_old, p_new, q_old, q_new))
            # else:
                # print("\nLoad at bus %i changed from %f to %f (real power).\n" % (loads.bus[i], p_old, p_new))
            loads.pf[i] = loads.p[i]/(np.sqrt(loads.p[i]**2 + loads.q[i]**2)) #update pf
            j += 1
            
    print("\nTotal variation in real power load: %f pu\n" % psi_load)
    
    system.update({'loads':loads})
    
    return


def gen_variation(system, gen_indices=np.array([]), scalings=np.array([])):
    #accepts an array of gen indices to scale and an array of the 
    #corresponding scaling factors
    #Scale of 1.0 = 100%, 1.10 = 110% (+10%) etc.
    
    gens = system.get('generators')
    psi_gen = 0
    
     
    j = 0   
    for i in gen_indices:
        p_old = gens.pset[i]
        p_new = p_old * scalings[j]
        gens.pset[i] = p_new
        psi_gen += p_new - p_old
        j += 1
            
    print("\nTotal variation in real power generation: %f pu\n" % psi_gen)
    
    system.update({'generators':gens})
    
    return
