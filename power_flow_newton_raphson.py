"""
Pseudocode:

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

...

"""


##Implementation

import numpy as np

def initialize_system(ybus, pset, qset, vmag_input = 0, custom_initial = False):
    ### INCOMPLETE

    #Read a file to construct ybus???

    n_buses = np.shape(ybus)[0] #read system size
    ymag = np.absolute(ybus)
    theta = np.angle(ybus)

    if custom_initial:
        1+1 #placeholder
        #load custom initial guesses
    else:
        delta = np.zeros((n_buses-1,1))
        vmag = np.ones((n_buses-1,1)) #this standard case should depend on whether there are more PV-busses than the slack bus?
        x = np.row_stack((delta,vmag))
    
    jacobian = np.zeros((2*(n_buses-1),2*(n_buses-1)))

    #Setup initial power mismatch vector delta_y
    p = np.zeros((n_buses-1,1))
    q = np.zeros((n_buses-1,1))

    for k in range(n_buses-1):
        psum = 0
        qsum = 0
        for n in range(n_buses-1):
            psum += ymag[k+1,n] * vmag[n] * np.cos(delta[k] - delta[n] - theta[k+1,n])
            qsum += ymag[k+1,n] * vmag[n] * np.sin(delta[k] - delta[n] - theta[k+1,n])
        p[k] = vmag[k] * psum
        q[k] = vmag[k] * qsum
    
    y = np.row_stack(((pset - p),(qset - q)))

    return n_buses, ymag, theta, jacobian, y, x 
    ###Something here is wrong - maybe the 0-indexing is handled wrong with respect to the omission of bus 1 in most calculations...
        





def calculate_Jacobian(n_buses, jacobian, x, adm_mat):
    ### INCOMPLETE
    
    #recalculate Jacobian matrix based on new iterative values

    #Slicing the jacobian in order to handle the submatrices
    j1 = jacobian[0:(n_buses-1),0:(n_buses-1)]
    j2 = jacobian[0:(n_buses-1),(n_buses-1):(2*(n_buses-1))]
    j3 = jacobian[(n_buses-1):(2*(n_buses-1)),0:(n_buses-1)]
    j4 = jacobian[(n_buses-1):(2*(n_buses-1)),(n_buses-1):(2*(n_buses-1))]

    #Diagonal elements

    #Off-diagonal elements