import numpy as np
import pandas as pd
import pandapower as pp

def load_pandapower_case(network):
    baseMVA = network.sn_mva #base power for system

    pp.runpp(network, enforce_q_lims=True) #run power flow
    ybus = network._ppc["internal"]["Ybus"].todense() #extract Ybus after running power flow
    gen = network.gen
    load = network.load
    slack = network.ext_grid
    buses = network.bus

    #Saving PandaPower results and per-unitizing power values
    pf_results = network.res_bus
    pf_results['p_pu'] = pf_results.p_mw/baseMVA
    pf_results['q_pu'] = pf_results.q_mvar/baseMVA
    
    slack_dict = {'bus':slack.bus[0], 'vset':slack.vm_pu[0], 'pmin':slack.min_p_mw[0]/baseMVA,
                  'pmax':slack.max_p_mw[0]/baseMVA, 'qmin':slack.min_q_mvar[0]/baseMVA, 
                  'qmax':slack.max_q_mvar[0]/baseMVA}
    #Setup system dictionary
    system = {'admmat':ybus,'slack':slack_dict, 'buses':[], 'generators':[],'loads':[],
              'iteration_limit':15,'tolerance':1e-3}

    #initializing empty lists
    gen_list = []
    load_list = []
    bus_list = []

    #Fill lists of generator and load dictionaries based on the loaded generator and load information from PandaPower
    #Per-unitizing the values according to the power base
    for i in range(len(gen.index)):
        gen_list.append({'type':'pv', 'bus':gen.bus[i], 'vset':gen.vm_pu[i], 'pset':gen.p_mw[i]/baseMVA,
                         'qset':None, 'qmin':gen.min_q_mvar[i]/baseMVA, 'qmax':gen.max_q_mvar[i]/baseMVA,
                         'pmin':gen.min_p_mw[i]/baseMVA, 'pmax':gen.max_p_mw[i]/baseMVA})

    for i in range(len(load.index)):
        load_list.append({'bus':load.bus[i], 'p':load.p_mw[i]/baseMVA, 'q':load.q_mvar[i]/baseMVA})
        
    for i in range(len(buses.index)):
        bus_list.append({'v_max':buses.max_vm_pu[i], 'v_min':buses.min_vm_pu[i], 'v_base':buses.vn_kv[i],
                         'zone':buses.zone[i], 'index':buses.name[i]})

    system.update({'generators':gen_list})
    system.update({'loads':load_list})
    system.update({'buses':bus_list})
    
    return system

def process_admittance_mat(system):
    ybus = system.get('admmat')
    n_buses = ybus.shape[0]
    g = np.real(ybus)
    b = np.imag(ybus)
    return n_buses, g, b

def get_pv_idx(system):
    
    pv_idx = np.empty((1,0), dtype=int)
    gens = system.get('generators')
    for gen in gens:
        if gen.get('type') == 'pv':
            pv_idx = np.append(pv_idx, gen.get('bus'))
    
    return pv_idx

def slack_idx(system):
    return system.get('slack').get('bus')

def init_voltage_vecs(system):
    ybus = system.get('admmat')
    n_buses = ybus.shape[0]
    #vectors containing voltage magnitude and angle information on all busses
    vmag_full = np.ones((n_buses,1))
    delta_full = np.zeros((n_buses,1))

    #setting slack bus voltage magnitude
    vmag_full[slack_idx(system)] = system.get('slack').get('vset')


    #Checking for PV-busses in order to simplify eventual calculations
    pv_idx = get_pv_idx(system)
    pv_slack_idx = np.sort(np.append(pv_idx, slack_idx(system))) #pv and slack indices
    vset = np.empty((1,0), dtype=int)
    gens = system.get('generators')
    for gen in gens:
        if gen.get('type') == 'pv':
            vset = np.append(vset, gen.get('vset'))
            vset = np.reshape(vset, (np.size(vset),1))
    if np.size(pv_idx) != 0:
        vmag_full[pv_idx] = vset
        vmag = np.delete(vmag_full, pv_slack_idx, 0)
    else:
        vmag = np.delete(vmag_full, slack_idx(system), 0)

    delta = np.delete(delta_full, slack_idx(system), 0)
    
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
    pv_slack_idx = np.sort(np.append(pv_idx, slack_idx(system))) #pv and slack indices

    if np.size(pv_idx) != 0:
        q = np.delete(q_full, pv_slack_idx, 0) #removing the pv bus indices after calculation
    else:
        q = np.delete(q_full, slack_idx(system), 0) #removing slack bus index 
        
    p = np.delete(p_full, slack_idx(system), 0)   
    
    return p, q, p_full, q_full


def calc_power_setpoints(system):
    ybus = system.get('admmat')
    n_buses = ybus.shape[0]
    
    pv_idx = get_pv_idx(system)
    pv_slack_idx = np.sort(np.append(pv_idx, slack_idx(system))) #pv and slack indices
    
    #loading bus setpoints
    pset = np.zeros((n_buses,1))
    qset = np.zeros((n_buses,1))
    
    gens = system.get('generators')
    loads = system.get('loads')
    
    for load in loads:
        k = load.get('bus')
        pset[k] -= load.get('p') #load is a negative injection
        qset[k] -= load.get('q')
    for gen in gens:
        k = gen.get('bus')
        pset[k] += gen.get('pset') #generator is a positive injection
        if gen.get('type') == 'pq':
            qset[k] += gen.get('qset')
    
    if np.size(pv_idx) != 0:
        qset = np.delete(qset, pv_slack_idx, 0)
    else:
        qset = np.delete(qset, slack_idx(system), 0)
    
    pset = np.delete(pset, slack_idx(system), 0)
    
    #removing slack bus index
    return pset, qset


def calc_mismatch_vecs(system, p, q):
    
    (pset, qset) = calc_power_setpoints(system)
    
    del_p = pset - p
    del_q = qset - q
    return del_p, del_q


####################################################
#old version of Jacobian calculation without support for arbitrary slack bus index

# def calc_jacobian(system, vmag, delta, g, b, p, q):
#     ybus = system.get('admmat')
#     n_buses = ybus.shape[0]
    
#     jacobian = np.zeros((2*(n_buses-1),2*(n_buses-1)))
    
#     #Pointing to the submatrices
#     j1 = jacobian[0:(n_buses-1),0:(n_buses-1)]
#     j2 = jacobian[0:(n_buses-1),(n_buses-1):(2*(n_buses-1))]
#     j3 = jacobian[(n_buses-1):(2*(n_buses-1)),0:(n_buses-1)]
#     j4 = jacobian[(n_buses-1):(2*(n_buses-1)),(n_buses-1):(2*(n_buses-1))]

#     #Calculating Jacobian matrix
#     for k in range(1,n_buses):
#         for n in range(1, n_buses):
#             if k == n: #diagonal elements
#                 j1[k-1,n-1] = -q[k] - b[k,k] * vmag[k]**2
#                 j2[k-1,n-1] = p[k] / vmag[k] + g[k,k] * vmag[k]
#                 j3[k-1,n-1] = p[k] - g[k,k] * vmag[k]**2
#                 j4[k-1,n-1] = q[k] / vmag[k] - b[k,k] * vmag[k]

#             else: #off-diagonal elements
#                 j1[k-1,n-1] = vmag[k] * vmag[n] * (g[k,n]*(np.sin(delta[k] - delta[n])) - b[k,n]*np.cos(delta[k] - delta[n]))
#                 j2[k-1,n-1] = vmag[k] * (g[k,n]*(np.cos(delta[k] - delta[n])) + b[k,n]*np.sin(delta[k] - delta[n]))
#                 j3[k-1,n-1] = -vmag[k] * vmag[n] * (g[k,n]*(np.cos(delta[k] - delta[n])) + b[k,n]*np.sin(delta[k] - delta[n]))
#                 j4[k-1,n-1] = vmag[k] * (g[k,n]*(np.sin(delta[k] - delta[n])) - b[k,n]*np.cos(delta[k] - delta[n]))

#     return jacobian


# def jacobian_calc_simplify(system, jacobian):
#     ybus = system.get('admmat')
#     n_buses = ybus.shape[0]
    
#     pv_idx = get_pv_idx(system) #reading indices of PV-busses
    
#     #simplifies jacobian matrix in the presence of PV-busses by deleting rows and columns
#     if np.size(pv_idx) != 0:
#         jacobian_calc = np.delete(jacobian, pv_idx + n_buses - 2, 0) #n - 2 because bus 1 is index 0 in the jacobian matrix
#         jacobian_calc = np.delete(jacobian_calc, pv_idx + n_buses - 2, 1) #and the submatrices are (n-1) * (n-1)
#     else:
#         jacobian_calc = jacobian
#     return jacobian_calc


#####################################################

def calc_jacobian(system, vmag_full, delta_full, g_full, b_full, p_full, q_full):
    ybus = system.get('admmat')
    n_buses = ybus.shape[0]
    
    jacobian = np.zeros((2*(n_buses-1),2*(n_buses-1)))
    
    #Pointing to the submatrices
    j1 = jacobian[0:(n_buses-1),0:(n_buses-1)]
    j2 = jacobian[0:(n_buses-1),(n_buses-1):(2*(n_buses-1))]
    j3 = jacobian[(n_buses-1):(2*(n_buses-1)),0:(n_buses-1)]
    j4 = jacobian[(n_buses-1):(2*(n_buses-1)),(n_buses-1):(2*(n_buses-1))]

    #Excluding slack bus of arbitrary location from calculations
    vmag = np.delete(vmag_full, slack_idx(system), 0)
    delta = np.delete(delta_full, slack_idx(system), 0)
    p = np.delete(p_full, slack_idx(system), 0)
    q = np.delete(q_full, slack_idx(system), 0)
    g = np.delete(g_full, slack_idx(system), 0)
    g = np.delete(g, slack_idx(system), 1)
    b = np.delete(b_full, slack_idx(system), 0)
    b = np.delete(b, slack_idx(system), 1)


    #Calculating Jacobian matrix
    for k in range(n_buses-1):
        for n in range(n_buses-1):
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

    return jacobian

def jacobian_calc_simplify(system, jacobian):
    ybus = system.get('admmat')
    n_buses = ybus.shape[0]
    
    pv_idx = get_pv_idx(system) #reading indices of PV-busses
    
    #adjusting indices for jacobian simplification according to slack bus placement
    #any bus index above the slack bus placement is (actual bus index - 1) due to the slack bus
    #already being omitted from the jacobian matrix
    if np.size(pv_idx) != 0:
        for i in range(np.size(pv_idx)):
            if pv_idx[i] > slack_idx(system):
                pv_idx[i] -= 1
                
        #simplifies jacobian matrix in the presence of PV-busses by deleting rows and columns    
        jacobian_calc = np.delete(jacobian, pv_idx + n_buses - 1, 0) #n - 2 because bus 1 is index 0 in the jacobian matrix
        jacobian_calc = np.delete(jacobian_calc, pv_idx + n_buses - 1, 1) #and the submatrices are (n-1) * (n-1)
    else:
        jacobian_calc = jacobian
    return jacobian_calc



def next_iteration(jacobian, vmag, delta, del_p, del_q):
    x = np.row_stack((delta, vmag))
    y = np.row_stack((del_p, del_q))
    x_next = x + np.matmul(np.linalg.inv(jacobian), y) #calculating next iteration
    delta_next = x_next[0:np.size(delta)]
    vmag_next = x_next[np.size(delta):]
    return delta_next, vmag_next
    


def check_convergence(y, threshold):
    #returns true if all indices in mismatch vector are below error threshold (tolerance)
    return np.all(np.absolute(y) < threshold) 


def check_pv_bus(system, n_buses, q_full):
    #check if PV bus reactive power is within specified limits
    #if not, set bus(es) to PQ at Q limit and return a bool to specify whether recalculation should be performed
    limit_violation = False
    
    
    #Only the generator outputs should be considered, so the generator loads must be subtracted
    #when checking the limit violation for reactive power!    
    q_loads = np.zeros((n_buses,1))
    for load in system.get('loads'):
        k = load.get('bus')
        q_loads[k] = -load.get('q')
    
    gens = system.get('generators')
    for gen in gens:
        if gen.get('type') == 'pv': #only considering PV-busses
            k = gen.get('bus')
            if (q_full[k] - q_loads[k]) < gen.get('qmin'):
                qset = gen.get('qmin')
                gen.update({'type':'pq', 'qset':qset})
                limit_violation = True
                break
            elif (q_full[k] - q_loads[k]) > gen.get('qmax'):
                qset = gen.get('qmax')
                gen.update({'type':'pq', 'qset':qset})
                limit_violation = True
                break
            system.update({'generators':gens})
    
    if limit_violation == True:
        print('Generator reactive power limit violated at bus %d.\nType set to PQ with generator reactive power setpoint of %.2f pu.\n' % (k, qset))
    
    return limit_violation


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
        
            for i in range(1, iteration_limit + 1):
                (delta, vmag) = next_iteration(jacobian_calc, vmag, delta, del_p, del_q)
                
                delta_full[non_slack_idx] = delta #updating voltage angles on all busses except slack
                vmag_full[pq_idx] = vmag #updating voltage magnitudes on non-slack and non-PV busses
                
                (p, q, p_full, q_full) = calc_power_vecs(system, vmag_full, delta_full, g, b)
        
                jacobian = calc_jacobian(system, vmag_full, delta_full, g, b, p_full, q_full)
        
                jacobian_calc = jacobian_calc_simplify(system, jacobian)
        
                (del_p, del_q) = calc_mismatch_vecs(system, p, q)
                
                y = np.row_stack((del_p, del_q))
        
        
                # print("\nIteration %d:\n" % i)
                # print("delta:\n",delta * 180/np.pi)
                # print("vmag:\n",vmag)
                # print("mismatch vector:\n", y)
                # print("Jacobian:\n", jacobian_calc)
                
                
                if check_convergence(y, tolerance):
                    recalculate = check_pv_bus(system, n_buses, q_full)
                    
                    if recalculate:
                        print('Recalculating power flow...\n')
                    else:
                        print("Power flow converged at %d iterations (tolerance of %f).\n" % (i, tolerance))
                        #print("Mismatch vector (P injections)\n", del_p)
                        #print("Mismatch vector (Q injections)\n", del_q)
                        print("\nTable of results (power values are injections):\n")
                        
                        typelist = ['' for i in range(n_buses)]
                        typelist[system.get('slack').get('bus')] = 'SLACK'
                    
                        for gen in system.get('generators'):
                            k = gen.get('bus')
                            typelist[k] = gen.get('type').upper()
                        
                        for i in range(n_buses):
                            if typelist[i] == '':
                                typelist[i] = 'PQ'
                        
                        d = {'vmag_pu':vmag_full.flatten(), 'delta_deg':delta_full.flatten()*180/np.pi, 'p_pu':p_full.flatten(), 'q_pu':q_full.flatten(), 'type':typelist}
                        df = pd.DataFrame(data=d, index = np.arange(n_buses))
                        df.index.name = 'bus'
                        print(df)
                    break
                
                elif i == iteration_limit:
                    print("Power flow did not converge after %d iterations (tolerance of %f).\n" % (i, tolerance))
                    return None
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
    
        for i in range(1, iteration_limit + 1):
            (delta, vmag) = next_iteration(jacobian_calc, vmag, delta, del_p, del_q)
            #Calculating initial power vectors
            
            delta_full[non_slack_idx] = delta #updating voltage angles on all busses except slack
            vmag_full[pq_idx] = vmag #updating voltage magnitudes on non-slack and non-PV busses
            
            (p, q, p_full, q_full) = calc_power_vecs(system, vmag_full, delta_full, g, b)
    
            jacobian = calc_jacobian(system, vmag_full, delta_full, g, b, p_full, q_full)
    
            jacobian_calc = jacobian_calc_simplify(system, jacobian)
    
            (del_p, del_q) = calc_mismatch_vecs(system, p, q)
            
            y = np.row_stack((del_p, del_q))
    
    
            # print("\nIteration %d:\n" % i)
            # print("delta:\n",delta * 180/np.pi)
            # print("vmag:\n",vmag)
            # print("mismatch vector:\n", y)
            # print("Jacobian:\n", jacobian_calc)
    
            if check_convergence(y, tolerance):
                print("Power flow converged at %d iterations (tolerance of %f).\n" % (i, tolerance))
                #print("Mismatch vector (P injections)\n", del_p)
                #print("Mismatch vector (Q injections)\n", del_q)
                print("\nTable of results (power values are injections):\n")
                typelist = ['' for i in range(n_buses)]
                typelist[system.get('slack').get('bus')] = 'SLACK'
            
                for gen in system.get('generators'):
                    k = gen.get('bus')
                    typelist[k] = gen.get('type').upper()
                
                for i in range(n_buses):
                    if typelist[i] == '':
                        typelist[i] = 'PQ'
                
                d = {'vmag_pu':vmag_full.flatten(), 'delta_deg':delta_full.flatten()*180/np.pi, 'p_pu':p_full.flatten(), 'q_pu':q_full.flatten(), 'type':typelist}
                df = pd.DataFrame(data=d, index = np.arange(n_buses))
                df.index.name = 'bus'
                print(df)
                #break
                break
            
            elif i == iteration_limit:
                print("Power flow did not converge after %d iterations (tolerance of %f).\n" % (i, tolerance))
                return None
    
    
    p_loss = np.sum(p_full) #this simple line should work due to power flows being injections
    #net_currents = calc_line_flows(system, vmag_full, n_buses)
    net_currents = 0
    
    
    results = {'bus_results':df, 'line_flows':net_currents, 'losses':p_loss, 'mismatches':y}
    
    return results   


def calc_current_base(system, bus_idx):
    #I_base = S_base / (sqrt(3) * V_base) 
    #where S_base is 3-phase and V_base is line-to-line
    bus = system.get('buses')[bus_idx]
    return (system.get('s_base') * 1e6) / (1.73205080757 * bus.get('v_base') * 1e3)


def calc_power_from_to(system, vmag, delta, from_idx, to_idx):
    #Note - vmag and delta should be the full vectors
    (n, g, b) = process_admittance_mat(system)
    i = from_idx
    j = to_idx
    
    p = vmag[i] ** 2 * g[i,j] - vmag[i] * vmag[j] * (
        g[i,j] * np.cos(delta[i] - delta[j]) + b[i,j] * np.sin(delta[i] - delta[j]))
    
    q = - (vmag[i] ** 2) * b[i,j] - vmag[i] * vmag[j] * (
        g[i,j] * np.sin(delta[i] - delta[j]) - b[i,j] * np.cos(delta[i] - delta[j]))

    s = np.sqrt(p**2 + q**2)
    
    return s, p, q


def calc_abcd_param(system, line_idx):
    line = system.get('lines')[line_idx]
    omega = 2*np.pi*system.get('frequency') #support for non-50Hz systems
    l = line.get('length') #get line length in km
    
    z = complex(line.get('r_per_km'), line.get('x_per_km'))
    y = complex(line.get('g_mu_s_per_km')*1e-6, omega*line.get('c_nf_per_km')*1e-9)
    Y = y*l
    Z = z*l
    
    if l <= 25: #short line 
        a = 1
        d = 1
        b = Z
        c = 0
    elif l <= 250: #medium line - nominal parameters
        a = 1 + (Y*Z)/2
        d = 1 + (Y*Z)/2
        b = Z
        c = Y * (1 + (Y*Z)/4)
        
    else: #long line - exact parameters
        gamma = np.sqrt(z*y)
        z_c = np.sqrt(z/y)
        a = np.cosh(gamma*l)
        d = np.cosh(gamma*l)
        b = z_c * np.sinh(gamma*l)
        c = (1/z_c) * np.sinh(gamma*l)
    
    return a, b, c, d


def calc_line_flows(system, vmag, n_buses):
    #Line flows: Current, real, reactive, apparent power at each end of lines
    #P_ft, Ptf, Q_ft, Q_tf, I_ft, I_tf, S_ft, S_tf
    #where ft = from/to and tf = to/from
    
    
    ybus = system.get('admmat')
    s_base = system.get('s_base')
    
    
    #Outlined approach:
    # DONE 1: Use the P_ij and Q_ij equations to evaluate power flows
    # DONE 2: Write a function to return ABCD-parameters of transmission line based on length
    # 3: Use ABCD parameters and knowledge of power flows to calculate current flows 
    #    (see group exercises from 31730)
    # 4: Find base current and convert per unit current flows to kA
    # 5: Construct dataframe for results
    
    
    #Note: P_ij and Q_ij gives a deviation for reactive power flow - see Teams
    
    
    
    #  |S| = sqrt(3) * |V_LL| * |I_L|
    
    
    
    # d = {'vmag_pu':vmag_full.flatten(), 'delta_deg':delta_full.flatten()*180/np.pi, 'p_pu':p_full.flatten(), 'q_pu':q_full.flatten(), 'type':typelist}
    # df = pd.DataFrame(data=d, index = np.arange(len(system.get('lines'))))
    # df.index.name = 'line'
    
    #return df
    
    pass 
