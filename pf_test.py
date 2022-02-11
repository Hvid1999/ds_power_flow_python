import power_flow_newton_raphson as pf
import numpy as np

##initialization
ybus = np.array([[complex(0,-10),complex(0,10)],[complex(0,10),complex(0,-10)]])
n_buses = ybus.shape[0]
#buses = [pf.Bus() for i in range(n_buses)]
#buses[0].type = 'slack'

pset = np.array([0,-2.0])
qset = np.array([0,-1.0])

(jacobian, y, x, g, b, p, q) = pf.initialize_system(n_buses, ybus, pset, qset)

#The test script calculates the initial parameters for the example in the lecture 7 slides
#next step is to implement the iterative solution


""" delta = np.zeros((n_buses,1))
vmag = np.ones((n_buses,1))
g = np.real(ybus)
b = np.imag(ybus)

#Setup initial power mismatch vector delta_y
p = np.zeros((n_buses,1))
q = np.zeros((n_buses,1))

for k in range(1,n_buses): #k ignores the first index which is the slack bus
    psum = 0
    qsum = 0
    for n in range(n_buses):
        psum += vmag[n] * (g[k][n]*(np.cos(delta[k] - delta[n])) + b[k][n]*np.sin(delta[k] - delta[n]))
        qsum += vmag[n] * (g[k][n]*(np.sin(delta[k] - delta[n])) - b[k][n]*np.cos(delta[k] - delta[n]))
    p[k] = vmag[k] * psum
    q[k] = vmag[k] * qsum

#ignoring the first indices in p and q (slack bus)...
y = np.row_stack(((pset[1:] - p[1:]),(qset[1:] - q[1:])))

#Constructing Jacobian consisting of the four (N-1) X (N-1) submatrices
jacobian = np.zeros((2*(n_buses-1),2*(n_buses-1)))
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
            j4[k-1][n-1] = vmag[k] * (g[k][n]*(np.sin(delta[k] - delta[n])) - b[k][n]*np.cos(delta[k] - delta[n])) """
print("J: \n", np.round(jacobian, 2))
print("J^-1: \n", np.round(np.linalg.inv(jacobian), 2))
print("y:\n", y)
