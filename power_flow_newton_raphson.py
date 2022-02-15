"""
Pseudocode:


FUNCTIONS:


MAYBE THE INITIALIZING FUNCTION SHOULD BE SPLIT INTO SEVERAL FUNCTIONS AND CALL EACH OF THEM:
    Function 1: Create Y_bus
    Function 2: Setup initial values and bus types with power limits
    Function 3: Initial iteration of power mismatch 
              (should account for whether bus is load or generation (+/- on pset/qset))
    Function 4: Setup Jacobian


def initialize_system(, custom_initial = False):
    Load system bus admittance matrix
    Define system size - N buses
    Load input data values for P, Q, V, delta
    Load system limits for real and reactive power
    Setup initial guesses - custom or standard (1.0 mag and 0.0 angle)
    Setup initial power mismatch vector delta_y
    Setup initial Jacobian matrix (dimensions)


def calculate_Jacobian(n_buses, jacobian, x, adm_mat):
    recalculate Jacobian matrix based on new iterative values

    Slicing the jacobian in order to handle the submatrices

    Diagonal elements

    Off-diagonal elements


def next_iteration():
    Invert the Jacobian - for now, faster methods are not necessary
    Solve for x(i+1)
    Return new x-vector

def check_convergence():
    check for convergence based on power mismatches and tolerance level
    return boolean

def pv_bus_check():
    calculate reactive power Q at PV-busses and check if within predefined limits
    if not, set bus to PQ at Q limit and compute bus voltage



CLASSES:

class Bus:
    Contains the following information:
        Bus type - slack, pv, pq
        Real power injection
        Reactive power injection
        Voltage magnitude
        Voltage phase angle

(...
class Generator:
    Contains the following information:
        Bus index placement
        Power setpoints
        Reactive power limits
        (slack participation factor)
...)

...

"""


##Implementation

import numpy as np

def load_adm_mat():
    pass #placeholder - should probably load from file
    #return ymag, theta

def setup_buses():
    #Setup initial values and bus types with power limits
    pass #create array of buses (class) from input data


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
                j2[k-1][n-1] = -vmag[k] * vmag[n] * (g[k][n]*(np.cos(delta[k] - delta[n])) + b[k][n]*np.sin(delta[k] - delta[n]))
                j3[k-1][n-1] = vmag[k] * (g[k][n]*(np.cos(delta[k] - delta[n])) + b[k][n]*np.sin(delta[k] - delta[n]))
                j4[k-1][n-1] = vmag[k] * (g[k][n]*(np.sin(delta[k] - delta[n])) - b[k][n]*np.cos(delta[k] - delta[n]))

def calculate_power_vecs(n_buses, vmag, delta, b, g, pv_idx):
    ###Note! (should probably account for whether bus is load or generation (+/- on pset/qset))

    #vectors with possibility for containing information about every bus
    p_full = np.zeros((n_buses,1))
    q_full = np.zeros((n_buses,1))
    
    #k ignores the first index which is the slack bus
    for k in range(1,n_buses): #k ignores the first index which is the slack bus
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


def check_pv_bus(buses, q, pv_recalc_flag=False):
    #check if PV bus reactive power is within specified limits
    #if not, set bus(es) to PQ at Q limit and compute bus voltage
    pass


def simplify_jacobian(n_buses, pv_idx, jacobian): 
    #simplifies jacobian matrix in the presence of PV-busses by deleting rows and columns
    if pv_idx.size != 0:
        jacobian_calc = np.delete(jacobian, pv_idx + n_buses - 2, 0) #n - 2 because bus 1 is index 0 in the jacobian matrix
        jacobian_calc = np.delete(jacobian_calc, pv_idx + n_buses - 2, 1) #and the submatrices are (n-1) * (n-1)
    else:
        jacobian_calc = jacobian
    return jacobian_calc

class Bus:
    ##INCOMPLETE
    
    #constructor
    def __init__(self, type='pq', pset=0, qset=0, vset=0, qlim=0, gen = False):
        self.type = type
        self.gen = gen #generator?
        self.pset = pset
        self.qset = qset

        if vset == 0:
            self.vset = 1.0
        else:
            self.vset = vset
        
        self.qlim = qlim
    
    def __repr__(self):
        s = "Type: " + self.type.upper() + "\n" 
        #Add other parameters
        return s
