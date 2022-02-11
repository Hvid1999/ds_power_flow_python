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


def next_x():
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

""" MAYBE THE INITIALIZING FUNCTION SHOULD BE SPLIT INTO SEVERAL FUNCTIONS:
    Function 1: Create Y_bus
    Function 2: Setup initial values and bus types with power limits
    Function 3: Setup Jacobian 
    Function 4: Initial iteration of power mismatch 
              (should account for whether bus is load or generation (+/- on pset/qset))
    """

def load_adm_mat():
    pass #placeholder - should probably load from file
    #return ymag, theta

def setup_buses():
    pass #create array of buses (class) from input data

def setup_jacobian(buses):
    n = len(buses)
    jacobian = np.zeros((2*(n-1),2*(n-1)))

    #Two nested for-loops... if k=n, diagonal... else off-diagonal


    return jacobian


def initialize_system(n_buses, ybus, pset, qset, vset = 0):
    ### INCOMPLETE

    #Note: pset = (P_G - P_L)
    g = np.real(ybus)
    b = np.imag(ybus)

    if vset != 0:
        1+1 #placeholder
        #load custom initial guesses
    else:
        delta = np.zeros((n_buses,1))
        vmag = np.ones((n_buses,1)) #this standard case should depend on whether there are more PV-busses than the slack bus?
        x = np.row_stack((delta,vmag))
    
    #Setup initial power mismatch vector delta_y
    p = np.zeros((n_buses,1))
    q = np.zeros((n_buses,1))

    #k ignores the first index which is the slack bus
    for k in range(1,n_buses): #k ignores the first index which is the slack bus
        psum = 0
        qsum = 0
        for n in range(n_buses):
            psum += vmag[n] * (g[k][n]*(np.cos(delta[k] - delta[n])) + b[k][n]*np.sin(delta[k] - delta[n]))
            qsum += vmag[n] * (g[k][n]*(np.sin(delta[k] - delta[n])) - b[k][n]*np.cos(delta[k] - delta[n]))
        p[k] = vmag[k] * psum
        q[k] = vmag[k] * qsum
    
    y = np.row_stack(((pset[1:] - p[1:]),(qset[1:] - q[1:])))

    #Setup initial jacobian
    jacobian = np.zeros((2*(n_buses-1),2*(n_buses-1)))
    calculate_jacobian(n_buses, jacobian, vmag, delta, g, b, p, q)

    return jacobian, y, x, g, b, p, q
    
        

def calculate_jacobian(n_buses, jacobian, vmag, delta, g, b, p, q):
       
    #recalculate Jacobian matrix based on new iterative values

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


class Bus:
    ##INCOMPLETE
    
    #constructor
    def __init__(self, type='pq', p=0, q=0, v=0, delta=0, gen = False):
        self.type = type
        self.pinj = p
        self.qinj = q
        self.vmag = v
        self.delta = delta  
        self.gen = gen #generator?
    
    def __repr__(self):
        s = "Type: " + self.type.upper() + "\n" 
        #Add other parameters
        return s
