#!/usr/bin/env python
## autograd+scipy demo for optimising curves

#%%
import autograd.numpy as np
from autograd import grad, jacobian, hessian
from scipy.optimize import minimize,NonlinearConstraint,LinearConstraint
import matplotlib.pyplot as plt

#%% squared sum of the segment length
def total_length(X):
    x=np.reshape(X,(-1,2))
    t=x[1:]-x[:-1]
    return np.sum(t*t)

# cos of adjacent tangents
def curvature(X):
    x=np.reshape(X,(-1,2))
    t=x[1:]-x[:-1]
    d2=np.sum(t*t, axis=1)
    K = 1-((t[1:,0]*t[:-1,0]+t[1:,1]*t[:-1,1])/np.sqrt(d2[1:]*d2[:-1]))
    return K

def squared_variation(u):
    return np.sum( (u[1:]-u[:-1])**2 )

# fix coords of end points, and y[N//3] > 1 and y[2N//3] < -1
def constraints(X):
    x=np.reshape(X,(-1,2))
    return np.array([x[0,0],x[0,1],x[N//3,1],x[2*N//3,1],x[N,0],x[N,1]])

def linear_combination_of_hessians(fun, argnum=0, *args, **kwargs):
    functionhessian = hessian(fun, argnum, *args, **kwargs)
    #not using wrap_nary_f because we need to do the docstring on our own
    def linear_combination_of_hessians(*funargs, **funkwargs):
        return np.tensordot(functionhessian(*funargs[:-1], **funkwargs), funargs[-1], axes=(0, 0))
    return linear_combination_of_hessians


#%% setup
N=9 # number of segments
weight = 0.1   #  try setting = 0.0

# target is the weighted sum of total length and curvature
target = lambda x: total_length(x) + weight*np.sum(curvature(x)**2)

# initial point
x0 = np.zeros((N+1,2))
x0[:,0] = np.linspace(0,1,N+1)
x0[1:-1,1] = np.random.uniform(-0.5,0.5,N-1)
X0 = x0.ravel()

# jacobian and hessian by autograd
jaco = jacobian(target)
hess = hessian(target)
constraints_hess = linear_combination_of_hessians(constraints)
constraints_jac = jacobian(constraints)

# non-linear inequality constraints
hard_constraint = NonlinearConstraint(constraints, [0,0,1,-np.inf,1,0],[0,0,np.inf,-1,1,0], jac=constraints_jac, hess=constraints_hess)

#%% optimise!
res = minimize(target, X0, method = 'trust-constr',
	       options={'xtol': 1e-10, 'gtol': 1e-8, 'disp': True, 'verbose': 1}, jac = jaco, hess=hess, constraints=[hard_constraint])

#%% plot result
plt.plot(x0[:,0],x0[:,1])
print("initial (blue):", target(x0),total_length(x0),np.sum(curvature(x0)**2))
x=np.reshape(res.x,(-1,2))
plt.plot(x[:,0],x[:,1])
print("optimised (orange):", target(x),total_length(x),np.sum(curvature(x)**2))

# %%
