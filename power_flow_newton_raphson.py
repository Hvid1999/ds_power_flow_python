import numpy as np

def load_system():
    pass #loads the relevant information - setpoints, busses, ybus etc.



def initialize_system(ybus, pset, qset, pv_idx, vset):
    ### INCOMPLETE

    #load_adm_mat()
    #setup_buses()

    #Note: pset = (P_G - P_L) 
    n_buses = ybus.shape[0]
    g = np.real(ybus)
    b = np.imag(ybus)

    #unknown variables vector
    vmag = np.ones((n_buses - 1 - np.size(pv_idx),1))
    delta = np.zeros((n_buses-1,1))

    #Setup initial power vectors
    p = np.zeros((n_buses - 1,1))
    del_p = np.zeros((n_buses - 1,1))
    q = np.zeros((n_buses - 1 - np.size(pv_idx),1))
    del_q = np.zeros((n_buses - 1 - np.size(pv_idx),1))

    #Setup initial jacobian
    jacobian = np.zeros((2*(n_buses-1),2*(n_buses-1)))

    #vectors containing voltage magnitude and angle information on all busses
    vmag_full = np.ones((n_buses,1))
    delta_full = np.zeros((n_buses,1))
    if np.size(pv_idx) != 0:
        vmag_full[pv_idx] = vset

    return jacobian, n_buses, vmag, delta, vmag_full, delta_full, g, b, p, q, del_p, del_q
    
        

def calculate_jacobian(n_buses, jacobian, vmag, delta, g, b, p, q):

    #Pointing to the submatrices
    j1 = jacobian[0:(n_buses-1),0:(n_buses-1)]
    j2 = jacobian[0:(n_buses-1),(n_buses-1):(2*(n_buses-1))]
    j3 = jacobian[(n_buses-1):(2*(n_buses-1)),0:(n_buses-1)]
    j4 = jacobian[(n_buses-1):(2*(n_buses-1)),(n_buses-1):(2*(n_buses-1))]

    #Calculating Jacobian matrix
    for k in range(1,n_buses):
        for n in range(1, n_buses):
            if k == n: #diagonal elements
                j1[k-1][n-1] = -q[k] - b[k][k] * vmag[k]**2
                j2[k-1][n-1] = p[k] / vmag[k] + g[k][k] * vmag[k]
                j3[k-1][n-1] = p[k] - g[k][k] * vmag[k]**2
                j4[k-1][n-1] = q[k] / vmag[k] - b[k][k] * vmag[k]

            else: #off-diagonal elements
                j1[k-1][n-1] = vmag[k] * vmag[n] * (g[k][n]*(np.sin(delta[k] - delta[n])) - b[k][n]*np.cos(delta[k] - delta[n]))
                j2[k-1][n-1] = vmag[k] * (g[k][n]*(np.cos(delta[k] - delta[n])) + b[k][n]*np.sin(delta[k] - delta[n]))
                j3[k-1][n-1] = -vmag[k] * vmag[n] * (g[k][n]*(np.cos(delta[k] - delta[n])) + b[k][n]*np.sin(delta[k] - delta[n]))
                j4[k-1][n-1] = vmag[k] * (g[k][n]*(np.sin(delta[k] - delta[n])) - b[k][n]*np.cos(delta[k] - delta[n]))

def simplify_jacobian(n_buses, pv_idx, jacobian): 
    #simplifies jacobian matrix in the presence of PV-busses by deleting rows and columns
    if pv_idx.size != 0:
        jacobian_calc = np.delete(jacobian, pv_idx + n_buses - 2, 0) #n - 2 because bus 1 is index 0 in the jacobian matrix
        jacobian_calc = np.delete(jacobian_calc, pv_idx + n_buses - 2, 1) #and the submatrices are (n-1) * (n-1)
    else:
        jacobian_calc = jacobian
    return jacobian_calc



def calculate_power_vecs(n_buses, vmag, delta, b, g, pv_idx):
    ###Note! (should probably account for whether bus is load or generation (+/- on pset/qset))

    #vectors with possibility for containing information about every bus
    p_full = np.zeros((n_buses,1))
    q_full = np.zeros((n_buses,1))
    
    #k ignores the first index which is the slack bus
    for k in range(n_buses): 
        psum = 0
        qsum = 0
        for n in range(n_buses):
            psum += vmag[n] * (g[k][n]*(np.cos(delta[k] - delta[n])) + b[k][n]*np.sin(delta[k] - delta[n]))
            qsum += vmag[n] * (g[k][n]*(np.sin(delta[k] - delta[n])) - b[k][n]*np.cos(delta[k] - delta[n]))
        p_full[k] = vmag[k] * psum
        q_full[k] = vmag[k] * qsum

    if np.size(pv_idx) != 0:
        q = np.delete(q_full[1:], pv_idx - 1, 0) #removing the pv bus indices after calculation
    else:
        q = q_full[1:]
    p = p_full[1:] #removing slack bus index
    return p, q, p_full, q_full




def update_mismatch_vector(p, q, pset, qset):
    del_p = pset - p
    del_q = qset - q
    return del_p, del_q



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


def validate_solution():
    pass #calculates power injections based on solution and compares to known values
    

def calc_line_flows():
    #Line flows: Current, real, reactive, apparent power at each end of lines
    #P_ft, Ptf, Q_ft, Q_tf, I_ft, I_tf, S_ft, S_tf
    #where ft = from/to and tf = to/from
    pass 


def check_pv_bus(pv_idx, pq_idx, qlim):
    #check if PV bus reactive power is within specified limits
    #if not, set bus(es) to PQ at Q limit and return a bool to specify whether recalculation should be performed
    pass


##############################################################################################
#CLASSES:
    
#CONSIDER DICTIONARIES FOR STORING AND UPDATING DATA

class Load:
    def __init__(self, bus=0, pset=0, qset=0, name=''):
        self.bus = bus
        self.p = pset
        self.q = qset
        self.name = name
    
    def __repr__(self):
        return "Load name: %s \nBus: %d \nP: %f \nQ: %f \n" % (self.name, self.bus, self.p, self.q)

class Generator:
   
    def __init__(self, typ='pv', bus = 0, pset=0, min_q=0, max_q=0, vset=0, name=''):
        self.type = typ
        self.bus = bus
        self.p = pset
        self.q_min = min_q
        self.q_max = max_q
        self.v = vset
        self.name = name
    
    def __repr__(self):
        if self.type.lower() == 'slack':
            return "Type: %s \nBus: %d \n" % (self.type.upper(), self.bus)
        else:
            return "Generator name: %s \nType: %s \nBus: %d \nP: %f \nQ limits: %f to %f \nV: %f \n" % (self.name, self.type.upper(), self.bus, self.p, self.q_min, self.q_max, self.v)
