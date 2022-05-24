#This function library was written as part of a bachelor thesis at the
#Technical University of Denmark (DTU) during the Spring semester of 2022.
#The code is meant to be easily readable and is built around the use of dictionaries
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
    baseMVA = network.sn_mva #power base for system
    freq = network.f_hz

    #run PandaPower power flow
    pp.runpp(network, enforce_q_lims = enforce_q_limits, trafo_model='pi', trafo_loading='power', max_iteration=25)
    #Saving PandaPower results and per-unitizing power values
    pandapower_results = network.res_bus
    pandapower_results['p_pu'] = pandapower_results.p_mw/baseMVA
    pandapower_results['q_pu'] = pandapower_results.q_mvar/baseMVA
    pandapower_results = pandapower_results[['vm_pu','va_degree','p_pu','q_pu']]

    ybus = network._ppc["internal"]["Ybus"].todense() #extract Ybus after running power flow
    ybus = np.asarray(ybus) #from matrix to array object
    gen = network.gen #voltage controlled generators
    sgen = network.sgen #static generators (PQ)
    load = network.load
    slack = network.ext_grid
    buses = network.bus #bus parameters
    lines = network.line #line parameters
    shunts = network.shunt #shunt elements for reactive power control
    trafo = network.trafo #transformers

    #Reformatting slack and generator dataframes and per-unitizing values
    slack = slack.rename(columns={'vm_pu':'vset', 'max_p_mw':'pmax', 'min_p_mw':'pmin',
                                  'max_q_mvar':'qmax', 'min_q_mvar':'qmin'})
    slack = slack[['in_service', 'bus', 'vset', 'pmax', 'pmin', 'qmax','qmin']]
    slack[['pmax', 'pmin', 'qmax', 'qmin']] = slack[['pmax', 'pmin','qmax', 'qmin']] / baseMVA

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
        #else, adding static generators to generator list as PQ-generators
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

    #Note! Shunts are handled as loads in this code.
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
        #Note: If a single slack bus is entered, the single slack power flow is obtained for that
        #generator.
        #Changing slack bus to PV-bus
        #The new P-setpoint of the original slack bus generator is the difference between 
        #total load and total generation unless a custom setpoint is entered
        
        if ref_bus_pset != 0:
            slack_pset = ref_bus_pset
        else:
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
        
        #Checking for entered slack generators and inactive generators
        for i in range(len((gens.index))):
            if np.size(slack_gens) != 0:
                if i in slack_gens:
                    gens.slack[i] = True
                else: 
                    gens.slack[i] = False
            if not gens.in_service[i]:
                gens.slack[i] = False
                    
        system.update({'generators':gens})
        
        if (np.size(slack_gens) == 1) and (slack.bus[0] != gens.bus[slack_gens[0]]) and not original_ref_bus:
            #If a single slack gen is specified, it is saved as the angle reference bus
            system.update({'reference_bus':system.get('generators').bus[slack_gens[0]]})
        else:
            #otherwise the original single slack bus is kept as angle reference
            system.update({'reference_bus':slack.bus[0]})
        
        del system['slack'] #removing separate slack bus description from system dictionary
        
        load_participation_factors(system, p_factors=participation_factors) #loading either equal p-factors or custom ones

    return (system, pandapower_results)


def load_participation_factors(system, p_factors = np.array([])):
    #accepts an array of participation factors ordered by increasing generator bus indices
    #if no array is entered, slack is distributed evenly among generators participating in slack
    
    #the size of the p-factor vector must be the number of controllable generators
    #the sum of the p-factors must be 1
    
    gens = system.get('generators')
    num_gens = len(gens.index)
    
    if np.size(p_factors) == 0: #standard case for no input - equal factors
        participation_factors = np.ones(num_gens)
        participation_factors = participation_factors / num_gens
        
    elif np.any(p_factors < 0):
        print('Error loading participation factors - all values must be non-negative.')
        print('Set to equal factors (standard case).\n')
        participation_factors = np.ones(num_gens)
        participation_factors = participation_factors / num_gens
    
    elif np.size(p_factors) != num_gens:
        print('Error loading participation factors - array length not equal to number of generators.')
        print('Set to equal factors (standard case).\n')
        participation_factors = np.ones(num_gens)
        participation_factors = participation_factors / num_gens
            
    elif round(sum(p_factors),3) != 1.0:
        #if the sum of the factors is not 1, the vector is normalized to enfore this.
        print('Error loading participation factors - sum (%f) not equal to 1.' % sum(p_factors))
        print('Input array normalized.\n')
        participation_factors = p_factors / np.sum(p_factors)
    else:
        participation_factors = p_factors.copy()
        
    
    #Checking validity of participation factors against generator status and slack participation
    for i in range(len(gens.index)):
        if (not gens.in_service[i]) and (participation_factors[i] != 0.0):
            print('Non-zero participation for inactive generator. Zero value enforced and array re-normalized.\n')
            participation_factors[i] = 0.0
            participation_factors = participation_factors / np.sum(participation_factors)
        elif (not gens.slack[i]) and (participation_factors[i] != 0.0):
            print('Non-zero participation for non-slack generator. Zero value enforced and array re-normalized.\n')
            participation_factors[i] = 0.0
            participation_factors = participation_factors / np.sum(participation_factors)
    
    #Loading the participation factors into the generator dataframe            
    for i in range(len(gens.index)):
        gens.participation_factor[i] = participation_factors[i]
    
    system.update({'generators':gens})
    return

def set_reference_bus_power(system, pset):
    #Function used to adjust the power setpoints of the reference bus,
    #which is often the original slack bus for converted single slack bus systems
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
    #the change makes little to no difference for power flow, but it may be more
    #accurate - for example when calculating currents in amperes.
    
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

    #Fixing erroneously defined transformer LV/HV sides
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

# =============================================================================
# Functions for extracting system information


def process_admittance_mat(system):
    #extract the number of buses and real and imaginary parts of the Ybus
    ybus = system.get('admmat')
    n_buses = ybus.shape[0]
    g = np.real(ybus)
    b = np.imag(ybus)
    return n_buses, g, b

def get_pv_idx(system):
    #returns an array containing the indices of PV-buses
    
    inactive_buses = inactive_bus_idx(system)
    gens = system.get('generators')
    pv_idx = gens[gens.in_service & (gens.type=='pv')].bus.to_numpy()
    #accounting for inactive buses effectively reducing the bus indices
    for i in range(np.size(inactive_buses)):
        pv_idx[pv_idx > inactive_buses[i]] -= 1
    
    return pv_idx

def slack_idx(system):
    #returns the bus index of the slack or reference bus of the system
    inactive_buses = inactive_bus_idx(system)
    
    if system.get('distributed_slack'):
        bus = system.get('reference_bus')
    else:
        bus = system.get('slack').bus[0]
    
    #accounting for inactive buses
    for i in range(np.size(inactive_buses)):
        if bus > inactive_buses[i]:
            bus -= 1
            
    return bus
                

def inactive_bus_idx(system):
    #returns an array of the indices of inactive buses in the system
    #this is used to allow adjust power flow calculations for inactive buses
    buses = system.get('buses')
    return buses[buses.in_service == False].index.to_numpy()


def build_admittance_matrix(system):
# =============================================================================
# UNFINISHED - does not match pandapower values for transformers in the matrix
#              also, it does currently not account for inactive buses
# =============================================================================
    
    #Calculates the bus admittance matrix based exclusively on t
    #ransmission line and transformer data.
    #line model: pi
    #trafo model: series inductance / pi (simple - no tap changing)
    
    #1 - calculate per unitized values according to base impedances
    #2 - build off-diagonals
    #3 - build diagonals
    
    
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
    
    
    #Delete rows and columns for inactive buses
    #easiest way is probably to delete rows and columns where the diagonal elements
    #are equal to zero, since this means that nothing is connected to the bus
    
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
    
    if np.size(pv_idx) != 0:
        #loading voltage setpoints for PV generators
        for i in range(len(gens.index)):
            if (gens.type[i] == 'pv') and gens.in_service[i]:
                vset = np.append(vset, gens.vset[i])
                
        vset = np.reshape(vset, (np.size(vset),1))
        vmag_full[pv_idx] = vset
    
    #simplifying vectors according to PV and slack buses
    if system.get('distributed_slack'):
        vmag = np.delete(vmag_full, pv_idx, 0) #removing known PV bus voltage magnitudes
    else:
        vmag_full[slack_idx(system)] = system.get('slack').vset[0]#setting slack bus voltage magnitude
        pv_slack_idx = np.sort(np.append(pv_idx, slack_idx(system))) #pv and slack indices
        vmag = np.delete(vmag_full, pv_slack_idx, 0) #removing slack bus and PV busses
    
    delta = np.delete(delta_full, slack_idx(system), 0) #reference voltage angle
    
    return vmag, delta, vmag_full, delta_full


def calc_power_vecs(system, vmag, delta, g, b):
    n_buses = g.shape[0] 
    
    # g = np.asarray(g)
    # b = np.asarray(b)
    
    #vectors containing power injection information about every bus
    p_full = np.zeros((n_buses,1))
    q_full = np.zeros((n_buses,1))
    
    # #calculate full power vectors based on the voltages of the system 
    # for k in range(n_buses): 
    #     psum = 0
    #     qsum = 0
    #     for n in range(n_buses):
    #         psum += vmag[n] * (g[k,n]*(np.cos(delta[k] - delta[n])) + b[k,n]*np.sin(delta[k] - delta[n]))
    #         qsum += vmag[n] * (g[k,n]*(np.sin(delta[k] - delta[n])) - b[k,n]*np.cos(delta[k] - delta[n]))
    #     p_full[k] = vmag[k] * psum
    #     q_full[k] = vmag[k] * qsum
        
    #rewritten for speed    
    for k in range(n_buses):         
        delta_k = np.ones((n_buses,1)) * delta[k]
        
        psum = vmag * (g[k,:].reshape((n_buses,1)) * (np.cos(delta_k - delta)) + b[k,:].reshape((n_buses,1)) * np.sin(delta_k - delta))
        
        qsum = vmag * (g[k,:].reshape((n_buses,1)) * (np.sin(delta_k - delta)) - b[k,:].reshape((n_buses,1)) * np.cos(delta_k - delta))
        
        p_full[k] = vmag[k] * np.sum(psum)
        q_full[k] = vmag[k] * np.sum(qsum)
    
    #Checking for PV-busses in order to simplify eventual calculations
    pv_idx = get_pv_idx(system)
    
    if system.get('distributed_slack'):
        q = np.delete(q_full, pv_idx, 0) #removing the pv bus indices
        p = p_full
    else:
        pv_slack_idx = np.sort(np.append(pv_idx, slack_idx(system))) #pv and slack indices
        q = np.delete(q_full, pv_slack_idx, 0) #removing the pv and slack bus indices after calculation
        p = np.delete(p_full, slack_idx(system), 0) #removing slack bus index from power vector
        
    return p, q, p_full, q_full


def calc_power_setpoints(system):
    n_buses = system.get('n_buses')
    
    inactive_buses = inactive_bus_idx(system)
    offset = j = 0
    
    pset = np.zeros((n_buses,1))
    qset = np.zeros((n_buses,1))
    # pset = np.zeros(n_buses)
    # qset = np.zeros(n_buses)
    
    gens = system.get('generators')
    loads = system.get('loads')
    shunts = system.get('shunts')
    #Shunts are as of now handled as loads 
    #this allows them to affect bus voltages
    
    
    #loading bus setpoints
    for i in range(len(loads.index)):
        if loads.in_service[i]:
            if (np.size(inactive_buses) > j) and (loads.bus[i] > inactive_buses[j]):
                #To account for inactive buses, which essentialy reduces the 
                #amount of buses in the system from a calculations standpoint
                offset += 1
                j += 1
            k = loads.bus[i] - offset
            #load is a negative power injection
            pset[k] -= loads.p[i] 
            qset[k] -= loads.q[i]
    
    offset = j = 0
    for i in range(len(shunts.index)):
        if shunts.in_service[i]:
            if (np.size(inactive_buses) > j) and (shunts.bus[i] > inactive_buses[j]):
                offset += 1
                j += 1
            k = shunts.bus[i] - offset
            #shunt values are consumption (load convention)
            pset[k] -= shunts.p[i] 
            qset[k] -= shunts.q[i]
    
    offset = j = 0
    for i in range(len(gens.index)):
        if gens.in_service[i]:
            if (np.size(inactive_buses) > j) and (gens.bus[i] > inactive_buses[j]):
                offset += 1
                j += 1
            k = gens.bus[i] - offset
            pset[k] += gens.pset[i] #generator is a positive injection
            if gens.type[i] == 'pq':
                qset[k] += gens.qset[i]
    
    #rewritten for speed
    # bus_idx = loads[loads.in_service == True].bus.to_numpy()
    # p_load = loads[loads.in_service == True].p.to_numpy()
    # q_load = loads[loads.in_service == True].q.to_numpy()
    
    # for i in range(np.size(inactive_buses)):
    #     bus_idx[bus_idx > inactive_buses[i]] -= 1
    # pset[bus_idx] -= p_load
    # qset[bus_idx] -= q_load
    
    # bus_idx = shunts[shunts.in_service == True].bus.to_numpy()
    # p_shunt = shunts[shunts.in_service == True].p.to_numpy()
    # q_shunt = shunts[shunts.in_service == True].q.to_numpy()
    
    # for i in range(np.size(inactive_buses)):
    #     bus_idx[bus_idx > inactive_buses[i]] -= 1
    # pset[bus_idx] -= p_shunt
    # qset[bus_idx] -= q_shunt
    
    # bus_idx = gens[gens.in_service == True].bus.to_numpy()
    # p_gen = gens[gens.in_service == True].pset.to_numpy()
    # q_gen = gens[gens.in_service == True].qset.to_numpy()
    
    # for i in range(np.size(inactive_buses)):
    #     bus_idx[bus_idx > inactive_buses[i]] -= 1
    # pset[bus_idx] += p_gen
    # q_gen[q_gen == None] = 0 #handling no specified q setpoint
    # qset[bus_idx] += q_gen.astype(float)
    
    pv_idx = get_pv_idx(system)
    if system.get('distributed_slack'):
        qset = np.delete(qset, pv_idx, 0) #removing PV bus indices
    else:
        pv_slack_idx = np.sort(np.append(pv_idx, slack_idx(system))) #pv and slack indices
        qset = np.delete(qset, pv_slack_idx, 0) #removing PV and slack bus indices
        pset = np.delete(pset, slack_idx(system), 0) #removing slack bus index
    
    # return pset.reshape((np.size(pset),1)), qset.reshape((np.size(qset),1))
    return pset, qset


def calc_mismatch_vecs(system, p, q):
    #calculates power mismatch vectors
    #(setpoints minus calculated power vectors)
    (pset, qset) = calc_power_setpoints(system)
    del_p = pset - p
    del_q = qset - q
    return del_p, del_q


def calc_jacobian(system, vmag, delta, g, b, p, q):
    #calculates the full jacobian matrix for the power flow iteration
    n_buses = system.get('n_buses')
    
    jacobian = np.zeros((2*(n_buses),2*(n_buses)))
    #Pointing to the submatrices
    j1 = jacobian[0:(n_buses),0:(n_buses)]
    j2 = jacobian[0:(n_buses),(n_buses):(2*(n_buses))]
    j3 = jacobian[(n_buses):(2*(n_buses)),0:(n_buses)]
    j4 = jacobian[(n_buses):(2*(n_buses)),(n_buses):(2*(n_buses))]

    #Calculating Jacobian matrix
    # for k in range(n_buses):
    #     for n in range(n_buses):
    #         if k == n: #diagonal elements
    #             j1[k,n] = -q[k] - b[k,k] * vmag[k]**2
    #             j2[k,n] = p[k] / vmag[k] + g[k,k] * vmag[k]
    #             j3[k,n] = p[k] - g[k,k] * vmag[k]**2
    #             j4[k,n] = q[k] / vmag[k] - b[k,k] * vmag[k]

    #         else: #off-diagonal elements
    #             j1[k,n] = vmag[k] * vmag[n] * (g[k,n]*(np.sin(delta[k] - delta[n])) - b[k,n]*np.cos(delta[k] - delta[n]))
    #             j2[k,n] = vmag[k] * (g[k,n]*(np.cos(delta[k] - delta[n])) + b[k,n]*np.sin(delta[k] - delta[n]))
    #             j3[k,n] = -vmag[k] * vmag[n] * (g[k,n]*(np.cos(delta[k] - delta[n])) + b[k,n]*np.sin(delta[k] - delta[n]))
    #             j4[k,n] = vmag[k] * (g[k,n]*(np.sin(delta[k] - delta[n])) - b[k,n]*np.cos(delta[k] - delta[n]))
    
    
    #rewritten for speed
    #using the power of numpy vectorization to avoid nested for loop
    #to ensure vector shapes align, reshape is used, 
    #and flatten at the end to create a flat one-dimensional array
    for k in range(n_buses):
        delta_k = np.ones((n_buses,1)) * delta[k]
        
        #off-diagonal elements
        j1[k,:] = (vmag[k] * vmag * (g[k,:].reshape((n_buses,1))*(np.sin(delta_k - delta)) - b[k,:].reshape((n_buses,1))*np.cos(delta_k - delta))).flatten()
        j2[k,:] = (vmag[k] * (g[k,:].reshape((n_buses,1))*(np.cos(delta_k - delta)) + b[k,:].reshape((n_buses,1))*np.sin(delta_k - delta))).flatten()
        j3[k,:] = (-vmag[k] * vmag * (g[k,:].reshape((n_buses,1))*(np.cos(delta_k - delta)) + b[k,:].reshape((n_buses,1))*np.sin(delta_k - delta))).flatten()
        j4[k,:] = (vmag[k] * (g[k,:].reshape((n_buses,1))*(np.sin(delta_k - delta)) - b[k,:].reshape((n_buses,1))*np.cos(delta_k - delta))).flatten()
    
        #diagonal elements
        j1[k,k] = -q[k] - b[k,k] * vmag[k]**2
        j2[k,k] = p[k] / vmag[k] + g[k,k] * vmag[k]
        j3[k,k] = p[k] - g[k,k] * vmag[k]**2
        j4[k,k] = q[k] / vmag[k] - b[k,k] * vmag[k]
        
        
    if system.get('distributed_slack'):
        #loading vector of participation factors from the system
        part_facts = np.zeros((2 * n_buses, 1)) #the full vector containing zeros for non-slack buses
        gens = system.get('generators')
        
        inactive_buses = inactive_bus_idx(system)
        offset = j = 0
        for i in range(len(gens.index)):
            #loading generator participation into the jacobian
            #and accounting for inactive buses
            if (np.size(inactive_buses) > j) and (gens.bus[i] > inactive_buses[j]):
                offset += 1
                j += 1
            k = gens.bus[i] - offset
            part_facts[k] = gens.participation_factor[i]
        
        jacobian = np.append(jacobian, part_facts, axis = 1)
        
    return jacobian

def jacobian_calc_simplify(system, jacobian):
    #simplifies the jacobian matrix for calculations if possible
    n_buses = system.get('n_buses')
    pv_idx = get_pv_idx(system) #indices of PV-busses
    row_remove = np.array([], dtype=int) #used to track which rows to remove
    col_remove = np.array([], dtype=int) #used to track which columns to remove

    if system.get('distributed_slack'):
        ref_idx = system.get('reference_bus')
        #voltage angle is assumed known at the reference bus
        #the corresponding column of J1 and J3 may therefore be removed
        col_remove = np.append(col_remove, ref_idx)
        
        #removing appropriate rows and columns for PV buses
        col_remove = np.append(col_remove, pv_idx + n_buses) #offset of n_buses to reach J2 and J4
        row_remove = np.append(row_remove, pv_idx + n_buses) #offset of n_buses to reach J3 and J4
        
    else: #single slack
        slack_index = slack_idx(system)
        
        #removing rows and columns related to the slack bus
        col_remove = np.append(col_remove, slack_index)
        col_remove = np.append(col_remove, slack_index + n_buses)
        row_remove = np.append(row_remove, slack_index)
        row_remove = np.append(row_remove, slack_index + n_buses)
        
        #PV bus simplification
        col_remove = np.append(col_remove, pv_idx + n_buses) #offset of n_buses to reach J2 and J4
        row_remove = np.append(row_remove, pv_idx + n_buses) #offset of n_buses to reach J3 and J4
    
    #Deleting rows and columns and returning simplified jacobian    
    jacobian_calc = np.delete(jacobian, row_remove, 0) 
    jacobian_calc = np.delete(jacobian_calc, col_remove, 1)
    
    return jacobian_calc

def next_iteration(dist_slack, jacobian, vmag, delta, k_g, del_p, del_q):
    #single function for both single and distributed slack
    #calculates the next iteration of the power flow for distributed slack
    #based on inversion of the jacobian and matrix multiplication
    x = np.row_stack((delta, vmag))
    if dist_slack:
        x = np.append(x, [[k_g]], axis = 0) #append the slack parameter to iteration vector
    
    y = np.row_stack((del_p, del_q))

    # x_next = x + np.matmul(np.linalg.inv(jacobian), y) #calculating next iteration
    x_next = x + np.linalg.solve(jacobian, y) #calculating next iteration
    
    #separating variables
    delta_next = x_next[0:np.size(delta)]
    if dist_slack:
        vmag_next = x_next[np.size(delta):(np.size(x_next) - 1)]
        k_g_next = x_next[-1][0]
    else:
        #k_g_next is always returned, but is 0 for single slack (unused)
        k_g_next = 0 
        vmag_next = x_next[np.size(delta):]
        
    return delta_next, vmag_next, k_g_next

def check_convergence(dist_slack, delta_next, vmag_next, delta, vmag, k_g_next, k_g, threshold):
    #single function for checking both single and distributed slack
    #returns true or false based on magnitude in change of iteration values for voltages
    #iteration step-based convergence criteria are most common, but it can also
    #be based on magnitudes of mismatch vectors
    x_next = np.row_stack((delta_next, vmag_next))
    x = np.row_stack((delta, vmag))
    if dist_slack:
        x = np.append(x, [[k_g]], axis = 0)
        x_next = np.append(x_next, [[k_g_next]], axis = 0)

    # checkvec = np.ones((x.shape))
    # for i in range(np.size(x)):
    #     if abs(x[i]) > 0: #avoid division by zero
    #         checkvec[i] = (x_next[i] - x[i])/x[i]
    
    #rewritten for speed
    #division by zero is handled by setting the index to = 1, not zero
    #as that would place the index value below the convergence threshold        
    checkvec = np.divide(((x_next-x)), x, out=np.ones_like(x_next), where=x!=0)
    
    return np.all(np.absolute(checkvec) < threshold)

def check_pv_bus(system, n_buses, q_full, print_results):
    #check if PV bus reactive power is within specified limits
    #if not, set bus(es) to PQ at Q limit and return a bool to specify 
    #whether recalculation should be performed
    limit_violation = False
    
    inactive_buses = inactive_bus_idx(system)
    #Only the generator outputs should be considered, so load at the generator bus must be subtracted
    #when checking the limit violation for reactive power!    
    q_loads = np.zeros((n_buses + np.size(inactive_buses),1))
    loads = system.get('loads')
    gens = system.get('generators')
    shunts = system.get('shunts')
    
    for i in range(len(loads.index)):
        if loads.in_service[i]:
            k = loads.bus[i]
            q_loads[k] -= loads.q[i]
        
    for i in range(len(shunts.index)):
        if shunts.in_service[i]:
            k = shunts.bus[i]
            q_loads[k] -= shunts.q[i]
    
    q_loads = np.delete(q_loads, inactive_buses, axis=0)
    offset = j = 0
    
    for i in range(len(gens.index)):
        if (gens.type[i] == 'pv') and (gens.in_service[i]): #only considering in service PV-busses
            if (np.size(inactive_buses) > j) and (gens.bus[i] > inactive_buses[j]):
                #adjusting for inactive buses
                offset += 1
                j += 1 
            k = gens.bus[i] - offset  
            
            #Checking for static generators on the PV gen bus 
            #(not relevant for New England system)
            sgen = gens[(gens.bus == gens.bus[i]) & (gens.type == 'pq')].reset_index(drop=True)
            if len(sgen.index) != 0:
                q_sgen = 0
                for j in range(len(sgen.index)):
                    q_sgen += sgen.qset[j]
                q_gen = q_full[k] - q_loads[k] - q_sgen
            else:
                q_gen = q_full[k] - q_loads[k]
            
            #if limit is violated, flag for recalculation and exit for-loop
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

    if limit_violation == True:
        system.update({'generators':gens})
        if print_results: #toggles printing of information
            print('Generator reactive power limit violated at bus %d (%f pu).\nType set to PQ with generator reactive power setpoint of %.2f pu.\n' % (gens.bus[i], q_gen, qset))
    
    return limit_violation

def run_power_flow(system, enforce_q_limits, print_results=True, print_bus_type=True):
    #runs the Newton-Raphson based power flow algorithm on a valid system dictionary
    
    iteration_limit = system.get('iteration_limit')
    tolerance = system.get('tolerance')
    dist_slack = system.get('distributed_slack')
    recalculate = True
    recalcs = 0
    
    if dist_slack:
        print("-------------------------------------------------------------")
        print("Calculating power flow (distributed slack bus)...\n")
    else:
        print("-------------------------------------------------------------")
        print("Calculating power flow (single slack bus)...\n")
    
    #while Q-limits are violated, power flow is recalculated according to adjustments
    #if limits are not enforced, recalculate is set to False upon convergence of power flow
    while recalculate == True: 
        (n_buses, g, b) = process_admittance_mat(system)
    
        (vmag, delta, vmag_full, delta_full) = init_voltage_vecs(system)
        
        k_g = k_g_next = 0.0 #used for distributed slack
    
        (p, q, p_full, q_full) = calc_power_vecs(system, vmag_full, delta_full, g, b)
    
        jacobian = calc_jacobian(system, vmag_full, delta_full, g, b, p_full, q_full)
    
        jacobian_calc = jacobian_calc_simplify(system, jacobian)
    
        (pset, qset) = calc_power_setpoints(system)
    
        # (del_p, del_q) = calc_mismatch_vecs(system, p, q)
        del_p = pset - p
        del_q = qset - q
        
    
        #obtaining list of non-PV and non-slack busses
        pv_idx = get_pv_idx(system)
        pq_idx = np.arange(n_buses)
        non_ref_idx = np.delete(pq_idx, slack_idx(system), 0)
        pq_idx = np.delete(pq_idx, pv_idx, 0)
        if not dist_slack:
            pq_idx = pq_idx[pq_idx != slack_idx(system)]
        
        gens = system.get('generators')
    
        for i in range(1, iteration_limit + 1):
            #iterates on power flow until convergence until at maximum 
            #number of iterations
            (delta_next, vmag_next, k_g_next) = next_iteration(dist_slack, jacobian_calc, 
                                                                      vmag, delta, k_g, del_p, del_q)
            
            if check_convergence(dist_slack, delta_next, vmag_next, delta, vmag, k_g_next, k_g, tolerance):
                #If power flow has converged
                if enforce_q_limits:
                    recalculate = check_pv_bus(system, n_buses, q_full, print_results)
                else:
                    recalculate = False
                
                if recalculate: 
                    #if Q limits are violated, restart calculations after adjustments
                    if print_results:
                        print('Recalculating power flow...\n')
                    break
                else:
                    print("Power flow converged at %d iterations (tolerance of %.12f).\n" % (i, tolerance))
                    
                    delta_full[non_ref_idx] = delta_next #updating voltage angles on all busses except ref/slack bus
                    vmag_full[pq_idx] = vmag_next #updating voltage magnitudes on non-PV busses (and non-slack)
                    if dist_slack: 
                        k_g = k_g_next
                        
                    #calculating final power vectors and mismatches
                    (p, q, p_full, q_full) = calc_power_vecs(system, vmag_full, delta_full, g, b)
                    del_p = pset - p
                    del_q = qset - q
                    
                    if print_bus_type:
                        #reading bus types into list for saving results
                        typelist = ['' for i in range(n_buses)]
                        
                        inactive_buses = inactive_bus_idx(system)
                        
                        if not dist_slack: 
                            #since the distributed slack bus has no explicit slack bus
                            offset = 0
                            for i in range(np.size(inactive_buses)):
                                if system.get('slack').bus[0] > inactive_buses[i]:
                                    offset += 1
                            typelist[system.get('slack').bus[0] - offset] = 'SLACK'
                        
                        offset = j = 0
                        for i in range(len(gens.index)):
                            if (np.size(inactive_buses) > j) and (gens.bus[i] >= inactive_buses[j]):
                                #account for inactive buses
                                offset += 1
                                j += 1
                            k = gens.bus[i] - offset 
                            typelist[k] = gens.type[i].upper()
                        
                        for i in range(n_buses):
                            if typelist[i] == '':
                                typelist[i] = 'PQ'       
                                
                        d = {'vmag_pu':vmag_full.flatten(), 'delta_deg':delta_full.flatten()*180/np.pi, 'p_pu':p_full.flatten(), 'q_pu':q_full.flatten(), 'type':typelist}
                    else:
                        d = {'vmag_pu':vmag_full.flatten(), 'delta_deg':delta_full.flatten()*180/np.pi, 'p_pu':p_full.flatten(), 'q_pu':q_full.flatten()}
                        
                    #saving results in dataframe
                    df = pd.DataFrame(data=d, index = np.arange(n_buses))
                    df.index.name = 'bus'
                break
            
            elif i == iteration_limit: #no convergence
                print("Power flow did not converge after %d iterations (tolerance of %.12f).\n" % (i, tolerance))
                return None

            delta_full[non_ref_idx] = delta_next 
            vmag_full[pq_idx] = vmag_next 
            
            delta = np.copy(delta_next)
            vmag = np.copy(vmag_next)
            if dist_slack: 
                k_g = k_g_next
            
            (p, q, p_full, q_full) = calc_power_vecs(system, vmag_full, delta_full, g, b)
    
            jacobian = calc_jacobian(system, vmag_full, delta_full, g, b, p_full, q_full)
    
            jacobian_calc = jacobian_calc_simplify(system, jacobian)
            
            if dist_slack:
                del_p = pset - (p + slack_distribution(system, k_g))
                del_q = qset - q
            else:
                del_p = pset - p
                del_q = qset - q
                # (del_p, del_q) = calc_mismatch_vecs(system, p, q)
            
        #Tracking how many times the while-loop has run to avoid endless loop
        #if the number of recalculations exceed the number of generators,
        #something has most likely gone wrong, since at that point, they should all
        #be set to PQ and thus nothing more can be done by the check_pv_bus function
        recalcs += 1
        if recalcs > (len(gens.index)): 
            print('\nError - endless loop. Calculation terminated.\n')
            break
          
    #Saving and exporting power flow results as dictionary
    vmag_res = pd.Series.to_numpy(df['vmag_pu'])
    delta_res = pd.Series.to_numpy(df['delta_deg']) * np.pi / 180

    inactive_buses = inactive_bus_idx(system)
    for i in range(np.size(inactive_buses)):
        #re-inserting inactive buses in results dataframe to reobtain
        #actual number of buses and illustrate results for inactive buses
        empty_row = pd.DataFrame({"vmag_pu": np.nan, "delta_deg": np.nan, "p_pu":0, "q_pu":0, "type":np.nan},index=[0])
        df = pd.concat([df.iloc[:inactive_buses[i]], empty_row, 
                        df.iloc[inactive_buses[i]:]]).reset_index(drop=True)
        
    p_loss = calc_system_losses(system, vmag_res, delta_res)    

    line_flows = calc_line_flows(system, df)
    
    trafo_flows = calc_transformer_loadings(system, df)
    
    if dist_slack:
        #Calculating slack distribution and accounting for inactive buses
        active_gen_indices = gens[gens.in_service].index.to_numpy()
        indices = np.arange(system.get('n_buses'))
    
        #reverse loop to get proper correction of bus indices
        for i in range(np.size(inactive_buses)-1, -1, -1):
            indices[indices >= inactive_buses[i]] += 1
        
        slack_distribution_df = pd.DataFrame(data={'p_pu':(-1)*slack_distribution(system, k_g).flatten()}, 
                                                index = indices)
        slack_distribution_df.index.name = 'bus'
    
        slack_distribution_df = slack_distribution_df.filter(items = gens['bus'].to_numpy()[active_gen_indices], axis = 0)
        slack_distribution_df['\u03C0'] = gens['participation_factor'].to_numpy()[active_gen_indices]
        
        #some pandapower systems have multiple generators at a single bus (in case of static gens)
        #the line below is a workaround to avoid showing multiple busses and too much slack
        slack_distribution_df = slack_distribution_df.groupby(level=0).mean()
        
        #export results as dictionary
        results = {'bus_results':df, 'line_flows':line_flows, 'total_losses_pu':p_loss, 'transformer_flows':trafo_flows,
                    'mismatches':(del_p, del_q), 'slack_distribution':slack_distribution_df}
    else:
        results = {'bus_results':df, 'line_flows':line_flows, 'transformer_flows':trafo_flows, 
                    'total_losses_pu':p_loss, 'mismatches':(del_p, del_q)}
        
        
    gen_res = get_generator_results(system, results)
    results.update({'generator_results':gen_res})
    
    if print_results: 
        if dist_slack:
            print("\nSlack (%f p.u.) distribution across slack generators:\n" % (-1*k_g))
            print(slack_distribution_df)
        #prints bus results to terminal
        print("\nTable of results (power values are injections):\n")
        print(df)
        
        #prints warnings about limit violations
        print("\nWarnings:\n")
        check_p_limits(system, results)
        check_q_limits(system, results)
        check_bus_voltage(system, results)
        check_line_trafo_loading(system, results)
     
    return results 
        

# =============================================================================
# Functions for evaluating power flow results

def get_generator_results(system, results):
    #NB: As of now, results only valid for system in which there are 
    #not more than one generator on a single bus
    
    p_gen = np.copy(results.get('bus_results').p_pu.to_numpy())
    q_gen = np.copy(results.get('bus_results').q_pu.to_numpy())
    shunts = system.get('shunts')
    gens = system.get('generators')
    loads = system.get('loads')

    #Calculating vector of generator outputs
    for i in range(len(loads.index)):
        k = loads.bus[i]
        p_gen[k] += loads.p[i] #removing the negative load injections from the power vector
        q_gen[k] += loads.q[i] 
    
    for i in range(len(shunts.index)):
        k = shunts.bus[i]
        q_gen[k] += shunts.q[i] 

    gen_buses = gens['bus'].to_numpy()
    vres = np.copy(results.get('bus_results').vmag_pu.to_numpy()[gen_buses])
    deltares = np.copy(results.get('bus_results').delta_deg.to_numpy()[gen_buses])
    
    #saving results to dataframe
    data = {'bus':gens.bus.copy(), 'p_pu':p_gen[gen_buses], 'q_pu':q_gen[gen_buses], 
            'vmag_pu':vres, 'delta_deg':deltares, 'pmax':gens.pmax.copy()}
    gen_res = pd.DataFrame(data=data, index = gens.index)
    return gen_res

def check_p_limits(system, results):
    #NB: As of now, results only valid for system in which there are 
    #not more than one generator on a single bus
    gens = system.get('generators')
    p_gen = results.get('generator_results').p_pu
    
    p_limits = np.copy(gens.pmax)

    for i in range(np.size(p_gen)):
        if gens.in_service[i]:
            if p_gen[i] > p_limits[i]:
                k = gens.bus[i]
                magnitude = p_gen[i] - p_limits[i]
                print("\nGenerator(s) real power limit exceeded at bus %i by %f pu.\n" 
                      % (k, magnitude))
    return


def check_q_limits(system, results):
    #Only relevant if reactive power limits are not enforced in the power flow
    
    #NB: As of now, results only valid for system in which there are 
    #not more than one generator on a single bus
    q_gen = results.get('generator_results').q_pu
    gens = system.get('generators')
    q_max = np.copy(gens.qmax)
    q_min = np.copy(gens.qmin)

    for i in range(np.size(q_gen)):
        if gens.in_service[i]:
            if round(q_gen[i],4) > round(q_max[i],4):
                magnitude = q_gen[i] - q_max[i]
                k = gens.bus[i]
                print("\nGenerator(s) reactive power upper limit exceeded at bus %i by %f pu.\n" 
                      % (k, magnitude))
            elif round(q_gen[i],4) < round(q_min[i],4):
                magnitude = abs(q_min[i] - q_gen[i])
                k = gens.bus[i]
                print("\nGenerator(s) reactive power lower limit exceeded at bus %i by %f pu.\n" 
                      % (k, magnitude))        
    return

def check_bus_voltage(system, results):
    bus = system.get('buses')
    vmag = results.get('bus_results').vmag_pu
    
    for i in range(len(bus.index)):
        if bus.in_service[i]:
            if vmag[i] > bus.max_vm_pu[i]:
                magnitude = vmag[i] - bus.max_vm_pu[i]
                print("\nBus voltage upper limit exceeded at bus %i by %f pu.\n" 
                      % (i, magnitude))
            elif vmag[i] < bus.min_vm_pu[i]:
                magnitude = bus.min_vm_pu[i] - vmag[i]
                print("\nBus voltage lower limit exceeded at bus %i by %f pu.\n" 
                      % (i, magnitude))
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
    #returns a voltage phasor
    return complex(vmag[bus]*np.cos(delta_rad[bus]),vmag[bus]*np.sin(delta_rad[bus]))


def calc_line_flows(system, bus_results):
    #Line flows: Current, real, reactive, apparent power at each end of lines
    #P_ft, Ptf, Q_ft, Q_tf, I_ft, I_tf, S_ft, S_tf
    #where ft = from/to and tf = to/from
    
    vmag = bus_results.vmag_pu.to_numpy()
    delta = bus_results.delta_deg.to_numpy() * np.pi / 180
    
    s_base = system.get('s_base')
    freq = system.get('frequency')
    
    lines = system.get('lines')
    n_lines = len(lines.index)

    #initializing empty arrays for storing data
    i_ft_pu = np.zeros(n_lines, dtype = complex)
    i_tf_pu = np.zeros(n_lines, dtype = complex)

    s_ft_pu = np.zeros(n_lines, dtype = complex)
    s_tf_pu = np.zeros(n_lines, dtype = complex)

    i_ka = np.zeros(n_lines, dtype = complex)
    fr = np.zeros(n_lines, dtype = int)
    to = np.zeros(n_lines, dtype = int)
    loading_percent = np.zeros(n_lines)
    
    fr = np.copy(lines['from'].to_numpy())
    to = np.copy(lines['to'].to_numpy()) 
    
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
            
            loading_percent[i] = (np.abs(i_ka[i]) / lines['ampacity_ka'][i]) * 100
        
        
    p_ft_pu = np.real(s_ft_pu)
    p_tf_pu = np.real(s_tf_pu)

    q_ft_pu = np.imag(s_ft_pu)
    q_tf_pu = np.imag(s_tf_pu)

    p_loss = p_ft_pu + p_tf_pu
        
    #saving results to dataframe
    d = {'from':lines['from'].to_numpy(),'to':lines['to'].to_numpy(),
         'loading_percent':loading_percent, 'i_ka':np.abs(i_ka), 'p_ft_pu':p_ft_pu, 'p_tf_pu':p_tf_pu, 
         'p_loss_pu':p_loss, 'q_ft_pu':q_ft_pu, 'q_tf_pu':q_tf_pu, 'i_ft_pu':np.abs(i_ft_pu), 
         'i_tf_pu':np.abs(i_tf_pu), 's_ft_pu':np.abs(s_ft_pu), 
         's_tf_pu':np.abs(s_tf_pu)}
    df = pd.DataFrame(data=d, index = np.arange(n_lines))
    df.index.name = 'line'
    
    return df


def calc_transformer_loadings(system, bus_results):
    #Note: Simplified representation of transformer as a series impedance between busses
    #(typical per-unit representation)

    vmag = bus_results.vmag_pu.to_numpy()
    delta = bus_results.delta_deg.to_numpy() * np.pi / 180
    
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
    
    lv = np.copy(trafo['lv_bus'].to_numpy())
    hv = np.copy(trafo['hv_bus'].to_numpy())
    
    #Adjusting HV/LV indices according to inactive buses
    #to avoid errors when using the Ybus matrix for flow calculations
    #since the Ybus is reduced for inactive buses
    if np.size(inactive_bus_idx(system)) > 0:
        inactive_buses = inactive_bus_idx(system)
        vmag = np.delete(vmag, inactive_buses)
        delta = np.delete(delta, inactive_buses)
        
        for i in range(np.size(inactive_buses)-1, -1, -1):
            lv[lv > inactive_buses[i]] -= 1
            hv[hv > inactive_buses[i]] -= 1
    
    for i in range(n_trafo):
        if trafo.in_service[i] == False:
            pass #do nothing - all zeros
        else:
            v_lv = get_phasor(vmag, delta, lv[i])
            v_hv = get_phasor(vmag, delta, hv[i])
            #loading the per unit series impedance from the admittance matrix
            x_t = 1 / (-1 * ybus[lv[i], hv[i]]) 
            
            i_lv_pu[i] = (v_lv - v_hv) / x_t
            i_hv_pu[i] = (v_hv - v_lv) / x_t
            i_lv_ka[i] = i_lv_pu[i] * s_base / (np.sqrt(3) * bus.vn_kv[lv[i]])
            i_hv_ka[i] = i_hv_pu[i] * s_base / (np.sqrt(3) * bus.vn_kv[hv[i]])
            
            s_lv_pu[i] = v_lv * np.conj(i_lv_pu[i])
            s_hv_pu[i] = v_hv * np.conj(i_hv_pu[i])
            
            s_mva = abs(max(s_lv_pu[i], s_hv_pu[i]) * s_base) #complex power in MVA
            
            #calculating loading percentage based on transformer power rating
            loading_percent[i] = (s_mva / trafo['s_rated'][i]) * 100
    
    d = {'lv':trafo['lv_bus'].to_numpy(),'hv':trafo['hv_bus'].to_numpy(),'loading_percent':loading_percent, 'p_lv_pu':np.real(s_lv_pu), 
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

    # for k in range(n_buses):
    #     losses += vmag[k] ** 2 * g[k,k]
    #     for n in range(k + 1, n_buses): #starts at n = k + 1 to avoid n == k as well as repeating behavior
    #         losses += 2 * vmag[k] * vmag[n] * g[k,n] * np.cos(delta[k] - delta[n]) 
    
    #rewritten for speed
    for k in range(n_buses):
        losses += vmag[k] ** 2 * g[k,k]
        delta_k = delta[k]#starts at n = k + 1 to avoid n == k as well as repeating behavior
        losses += np.sum(2 * vmag[k] * vmag[(k+1):] * g[k,(k+1):] * np.cos(delta_k - delta[(k+1):])) 
    
    return losses

def slack_distribution(system, k_g):
    #returns a vector of the distribution of system slack across buses
    gens = system.get('generators')
    slackvec = np.zeros((system.get('n_buses'), 1))
    inactive_buses = inactive_bus_idx(system)
    offset = j = 0
    
    for i in range(len(gens.index)):
        if gens.slack[i]:
            if (np.size(inactive_buses) > j) and (gens.bus[i] > inactive_buses[j]):
                offset += 1
                j += 1
            k = gens.bus[i] - offset 
            p_fact = gens.participation_factor[i]
            #k_g is a negative injection, but the absolute value is taken here
            #because the vector denotes how much each slack generator injects
            #to compensate for losses
            slackvec[k] = p_fact * k_g
    
    return slackvec


# =============================================================================
# Functions for plotting results: Bus voltages, line/trafo loadings


def plot_results(system, results, angle=False ,name='', save_directory='', plot=''):
    #Note: need small changes if the system does not have transformers 
    #or if the system has different bus voltage limits for each bus
    #but the functionality is based on the New England 39 bus system
    
    if plot == 'lines':
        fig = plt.figure(dpi=200)
        fig.set_figheight(11)
        fig.set_figwidth(11)
        plt.bar(results.get('line_flows').index, results.get('line_flows')['loading_percent'], 
                    color='teal')
        # plt.scatter(results.get('line_flows').index, np.ones(len(results.get('line_flows').index))*100, marker="_", color='tab:red',s=30)
        plt.axhline(y=100, color='tab:red', linestyle='--')
        if  max(results.get('line_flows')['loading_percent']) > 100:
            plt.ylim(0,max(results.get('line_flows')['loading_percent']) + 5)
        else:
            plt.ylim(0,110)
        plt.title('Line Loading')
        plt.ylabel('Percentage')
        plt.xlabel('Line')
        plt.xticks(range(0, len(results.get('line_flows').index), 2))
        plt.grid(linestyle='--', linewidth=0.5, alpha=0.65)
        plt.margins(x=0.025)
        if name != '':
            plt.title('%s\n\n' % name, fontweight='bold', fontsize=14)
    elif plot == 'generators':
        fig = plt.figure(dpi=200)
        fig.set_figheight(11)
        fig.set_figwidth(11)
        gen_loadings = (results.get('generator_results')['p_pu'].to_numpy()/system.get('generators')['pmax'].to_numpy())*100
        plt.bar(results.get('generator_results').index, 
                gen_loadings, 
                color='darkcyan')
        # plt.scatter(results.get('line_flows').index, np.ones(len(results.get('line_flows').index))*100, marker="_", color='tab:red',s=30)
        plt.axhline(y=100, color='tab:red', linestyle='--')
        if  max(gen_loadings) > 100:
            plt.ylim(0,max(gen_loadings + 5))
        else:
            plt.ylim(0,110)
        plt.title('Generator Loading (P)')
        plt.ylabel('Percentage')
        plt.xlabel('Generator')
        plt.xticks(range(0, np.size(gen_loadings), 2))
        plt.grid(linestyle='--', linewidth=0.5, alpha=0.65)
        plt.margins(x=0.025)
        if name != '':
            plt.title('%s\n\n' % name, fontweight='bold', fontsize=14)
    elif plot == 'lg': #lines and generators
        gs = gsp.GridSpec(2, 1)
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
        gen_loadings = (results.get('generator_results')['p_pu'].to_numpy()/system.get('generators')['pmax'].to_numpy())*100
        ax1.bar(results.get('generator_results').index, gen_loadings, color='midnightblue')
        ax1.axhline(y=100, color='tab:red', linestyle='--')
        ax1.title.set_text('Generator Loading (P)')
        ax1.set_ylabel('Percentage')
        ax1.set_xlabel('Generator')
        ax1.set_xticks(range(0, len(gen_loadings), 2))
        ax1.grid(linestyle='--', linewidth=0.5, alpha=0.65)
        ax1.margins(x=0.025)
        
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
        
    else:
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
        ax1.set_xticks(range(0, len(results.get('bus_results').index), 2))
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
            ax4.set_xticks(range(0, len(results.get('bus_results').index), 2))
            ax4.grid(linestyle='--', linewidth=0.5, alpha=0.65)
            ax4.margins(x=0.025)
            ax4.set_ylim(-(max(abs(results.get('bus_results')['delta_deg']))+1), 
                         max(abs(results.get('bus_results')['delta_deg']))+1)
    
    if save_directory != '':
        fig.savefig(save_directory)
    return

def plot_result_comparison(results1, results2, angle=False, name = '', fixed_y_axis_values=[0,0,0,0]):
    #Plots differences for bus voltages, line loadings etc. between two result dictionaries
    #The fixed y axis values list indices correspond to the following plot y limits:
    #0: voltage magnitude, 1: line loading, 2: transformer loading, 3: voltage angle
    #The same limit is set for positive and negative values
    
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
    if fixed_y_axis_values[0] == 0:
        ax1.set_ylim(-max(abs(vmag_diff)), max(abs(vmag_diff)))
    else:
        ax1.set_ylim(-fixed_y_axis_values[0], fixed_y_axis_values[0])
    
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
    if fixed_y_axis_values[1] == 0:
        ax2.set_ylim(-max(abs(l_diff)), max(abs(l_diff)))
    else:
        ax2.set_ylim(-fixed_y_axis_values[1], fixed_y_axis_values[1])
    
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
    if fixed_y_axis_values[2] == 0:
        ax3.set_ylim(-max(abs(t_diff)), max(abs(t_diff)))
    else:
        ax3.set_ylim(-fixed_y_axis_values[2], fixed_y_axis_values[2])
    
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
        if fixed_y_axis_values[3] == 0:
            ax4.set_ylim(-max(abs(delta_diff)), max(abs(delta_diff)))
        else:
            ax4.set_ylim(-fixed_y_axis_values[3], fixed_y_axis_values[3])
        
    
    return


# =============================================================================
# Functions used for thesis test cases

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


def panda_disable_bus(network, bus_idx):
    #Fast track function to disable all network elements associated with the given bus
    #of a pandapower network
    line = network.line
    trafo = network.trafo
    gen = network.gen
    load = network.load
    shunt = network.shunt
    
    line_idx = line[(line.from_bus == bus_idx) | (line.to_bus == bus_idx)].index.tolist()
    trafo_idx = trafo[(trafo.lv_bus == bus_idx) | (trafo.hv_bus == bus_idx)].index.tolist()
    gen_idx = gen[gen.bus == bus_idx].index.tolist()
    load_idx = load[load.bus == bus_idx].index.tolist()
    shunt_idx = shunt[shunt.bus == bus_idx].index.tolist()
    
    line.in_service[line_idx] = False
    trafo.in_service[trafo_idx] = False
    gen.in_service[gen_idx] = False
    load.in_service[load_idx] = False
    shunt.in_service[shunt_idx] = False
    network.bus.in_service[bus_idx] = False
    return


def line_loading_metric_old(results_list):
    #array of zeros of length equal to amount of lines, 
    #assuming that all results are from the same system
    phi_lines = np.zeros(len(results_list[0].get('line_flows').index))
    
    
    for j in range(len(results_list)):
        line_flows = results_list[j].get('line_flows').copy()
        
        #computing the metric for line l
        for l in range(np.size(phi_lines)):
            phi_lines[l] += (line_flows['loading_percent'][l] / 100) ** 10
        
        #averaging across all results/contingencies
        if j == len(results_list):
            phi_lines = phi_lines / j
    
    phi = np.sum(phi_lines) / np.size(phi_lines)
    
    return phi

def line_loading_metric(results):
    #array of zeros of length equal to amount of lines
    phi_lines = np.zeros(len(results.get('line_flows').index))
    
    line_flows = results.get('line_flows').copy()
    
    #computing the metric for line l
    #raised to the power of 10 to strongly penalize lines close to 100% loading
    for l in range(np.size(phi_lines)):
        phi_lines[l] += (line_flows['loading_percent'][l] / 100) ** 30
    
    #averaging over all lines
    phi = np.sum(phi_lines) / np.size(phi_lines)
    
    return phi

def generator_limit_metric(system, results):
    
    gens = system.get('generators').copy()
    gen_res = results.get('generator_results').copy()
    #array of zeros of length equal to amount of gens
    phi_gens = np.zeros(len(gen_res.index))
    
    #computing the metric for line l
    #raised to the power of 10 to strongly penalize lines close to 100% loading
    for g in range(len(gen_res.index)):
        if gens.in_service[g]:
            phi_gens[g] += (gen_res['p_pu'][g] / gens['pmax'][g]) ** 30
    
    #averaging over all lines
    phi = np.sum(phi_gens) / np.size(phi_gens)
    
    return phi