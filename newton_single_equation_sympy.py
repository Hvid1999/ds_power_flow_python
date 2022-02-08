import sympy as sp

#accepts a symbolic equation (sympy) y(x) as input
#x_0 = initial guess
#i_max = maximum number of iterations
#tolerance = tolerance level for residual error
def single_newton_raphson(y, x_0, i_max, tolerance):
    yprime = y.diff(x)
    x_current = x_0
    for i in range(1, i_max + 1):
        x_new = x_current - y.subs(x,x_current)/yprime.subs(x,x_current)
        res = abs(x_new - x_current)/x_current
        print(i, float(x_new), float(res))
        x_current = x_new
        if res < tolerance: 
            print("The root is %f at %d iterations." % (x_new, i))
            return x_new, i
        elif i == i_max:
            print("No convergence at %d iterations." % i)
            return 0, 0

x = sp.Symbol('x')
y = x**2 - 2

(root, iterations) = single_newton_raphson(y, 3, 15, 0.001)


#For visualization
#sp.plot(y,(x,-5,5))
