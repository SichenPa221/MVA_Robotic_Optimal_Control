'''
Batch iLQR with computation of the manipulator dynamics

Copyright (c) 2023 Idiap Research Institute <https://www.idiap.ch/>
Written by Amirreza Razmjoo <amirreza.razmjoo@idiap.ch> and 
Sylvain Calinon <https://calinon.ch>

This file is part of RCFS <https://robotics-codes-from-scratch.github.io/>
License: MIT
'''

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches

# Helper functions
# ===============================

# Logarithmic map for R^2 x S^1 manifold
def logmap(f,f0):
    position_error = f[:,:2] - f0[:,:2]
    orientation_error = np.imag(np.log( np.exp( f0[:,-1]*1j ).conj().T * np.exp(f[:,-1]*1j).T )).conj().reshape((-1,1))
    error = np.hstack(( position_error , orientation_error ))
    return error

# Forward kinematics for E-E
def fkin(x,param):
	L = np.tril(np.ones([param.nbVarX, param.nbVarX]))
	f = np.vstack([
		param.l @ np.cos(L @ x.T),
		param.l @ np.sin(L @ x.T),
		np.mod(np.sum(x.T,0)+np.pi, 2*np.pi) - np.pi
	]) # f1,f2,f3, where f3 is the orientation (single Euler angle for planar robot)
	return f.T

# Forward Kinematics for all joints
def fkin0(x,param):
	L = np.tril(np.ones([param.nbVarX, param.nbVarX]))
	f = np.vstack([
		L @ np.diag(param.l) @ np.cos(L @ x),
		L @ np.diag(param.l) @ np.sin(L @ x)
	])
	f = np.hstack([np.zeros([2,1]), f])
	return f.T

# Jacobian with analytical computation (for single time step)
def Jkin(x,param):
    L = np.tril( np.ones((len(x),len(x))) )
    J = np.vstack((
        -np.sin(L@x).T @ np.diag(param.l) @ L,
        np.cos(L@x).T @ np.diag(param.l) @ L,
        np.ones(len(x))
    ))
    return J

# Residual and Jacobian
def f_reach(x,param):
    f = logmap(fkin(x,param),param.Mu)
    J = np.zeros(( param.nbPoints  * param.nbVarF , param.nbPoints  * param.nbVarX ))

    for t in range(param.nbPoints ):
        f[t,:2] = param.A[t].T @ f[t,:2] # Object oriented fk
        Jtmp = Jkin(x[t],param)
        Jtmp[:2] = param.A[t].T @ Jtmp[:2] # Object centered jacobian

        if param.useBoundingBox:
            for i in range(2):
                if abs(f[t,i]) < param.sz[i]:
                    f[t,i] = 0
                    Jtmp[i]=0
                else:
                    f[t,i] -=  np.sign(f[t,i]) * param.sz[i]
        J[ t*param.nbVarF:(t+1)*param.nbVarF , t*param.nbVarX:(t+1)*param.nbVarX] = Jtmp
    return f,J

# Forward dynamic to compute
def forward_dynamics(x, u, param):
    
    l = np.reshape( param.l, [1,param.nbVarX] )
    m = np.reshape( param.linkMasses, [1,param.nbVarX] )
    
    Lm = np.multiply(np.triu(np.ones([param.nbVarX,param.nbVarX])), np.repeat(m, param.nbVarX,0)) 
    L = np.tril(np.ones([param.nbVarX, param.nbVarX]))
    Su = np.zeros([2 * param.nbVarX * param.nbData , param.nbVarX * (param.nbData  - 1)])
    
    #Precomputation of mask (in tensor form)
    S1= np.zeros([param.nbVarX, param.nbVarX, param.nbVarX])
    J_index = np.ones([1, param.nbVarX])
    for j in range(param.nbVarX):
        J_index[0,:j] = np.zeros([j])
        S1[:,:,j] = np.repeat(J_index @ np.eye(param.nbVarX), param.nbVarX, 0) - np.transpose(np.repeat(J_index @ np.eye(param.nbVarX), param.nbVarX, 0))
     
    #Initialization of dM and dC tensors and A21 matrix
    dM = np.zeros([param.nbVarX, param.nbVarX, param.nbVarX])
    dC = np.zeros([param.nbVarX, param.nbVarX, param.nbVarX])
    A21 = np.zeros([param.nbVarX, param.nbVarX])
    
    for t in range(param.nbData-1):        
        
        # Computation in matrix form of G,M, and C
        G = np.reshape(np.sum(Lm,1), [param.nbVarX,1]) * l.T * np.cos(L @ np.reshape(x[t,0:param.nbVarX], [param.nbVarX,1])) * param.g
        G = L.T @ G
        C = (l.T * l) * np.sin(np.reshape(L @ x[t,:param.nbVarX], [param.nbVarX,1]) - L @ x[t,:param.nbVarX]) * (Lm ** .5 @ ((Lm ** .5).T))
        C = L.T @ C 
        M = (l.T * l) * np.cos(np.reshape(L @ x[t,:param.nbVarX], [param.nbVarX,1]) - L @ x[t,:param.nbVarX]) * (Lm ** .5 @ ((Lm ** .5).T))
        M = L.T @ M @ L 
        
        # Computation in tensor form of derivatives dG,dM, and dC
        dG_tmp = -np.diagflat(np.reshape(np.sum(Lm,1), [param.nbVarX,1]) * l.T * np.sin(L @ np.reshape(x[t,0:param.nbVarX], [param.nbVarX,1])) * param.g) @ L
        dG = L.T @ dG_tmp

        dM_tmp = (l.T * l) * np.sin(np.reshape(L @ x[t,:param.nbVarX], [param.nbVarX,1]) - L @ x[t,:param.nbVarX]) * (Lm ** .5 @ ((Lm ** .5).T)) 
        
        for j in range(dM.shape[2]):
            dM[:,:,j] = L.T @ (dM_tmp * S1[:,:,j]) @ L
        
        dC_tmp = -(l.T * l) * np.cos(np.reshape( L @ x[t,:param.nbVarX], [param.nbVarX,1]) - L @ x[t,:param.nbVarX]) * (Lm ** .5 @ ((Lm ** .5).T)) 
        
        for j in range(dC.shape[2]):
            dC[:,:,j] = L.T @ (dC_tmp * S1[:,:,j])

        # update pose 
        tau = np.reshape(u[(t) * param.nbVarX:(t + 1) * param.nbVarX], [param.nbVarX, 1])
        inv_M = np.linalg.inv(M)
        ddx = inv_M @ (tau - G - C @ (L @ np.reshape(x[t,param.nbVarX:], [param.nbVarX,1])) ** 2) - L @ np.reshape(x[t,param.nbVarX:], [param.nbVarX,1]) * param.kv
        
        # compute local linear systems
        x[t+1,:] = x[t,:] + np.hstack([x[t,param.nbVarX:], np.reshape(ddx, [param.nbVarX,])]) * param.dt
        A11 = np.eye(param.nbVarX)
        A12 = A11 * param.dt
        A22 = np.eye(param.nbVarX) + (-2 * inv_M @ C @ np.diagflat(L @ x[t,param.nbVarX:]) @ L - L * param.kv) * param.dt
        for j in range(param.nbVarX):
            
            A21[:,j] = (-inv_M @ dM[:,:,j] @ inv_M @ (tau - G - C @ (L @ np.reshape(x[t,param.nbVarX:], [param.nbVarX,1])) ** 2) 
                        - np.reshape(inv_M @ dG[:,j], [param.nbVarX,1]) 
                        - inv_M @ dC[:,:,j] @ (L @ np.reshape(x[t,param.nbVarX:], [param.nbVarX,1])) ** 2).flatten()
        A = np.vstack((np.hstack((A11, A12)), np.hstack((A21 * param.dt, A22))))
        B = np.vstack((np.zeros([param.nbVarX, param.nbVarX]), inv_M * param.dt))
        
        # compute transformation matrix
        Su[2 * param.nbVarX * (t + 1):2 * param.nbVarX * (t + 2),:] = A @ Su[2 * param.nbVarX * t:2 * param.nbVarX * (t + 1),:]
        Su[2 * param.nbVarX * (t + 1):2 * param.nbVarX * (t + 2), param.nbVarX * t:param.nbVarX * (t + 1)] =B
    return x, Su

# Parameters
# ===============================

param = lambda: None # Lazy way to define an empty class in python
param.dt = 1e-2 # Time step length
param.nbData = 50 # Number of datapoints
param.nbIter = 100 # Maximum number of iterations for iLQR
param.nbPoints = 2 # Number of viapoints
param.nbVarX = 3 # State space dimension (x1,x2,x3)
param.nbVarU = 3 # Control space dimension (dx1,dx2,dx3)
param.nbVarF = 3 # Objective function dimension (f1,f2,f3, with f3 as orientation)
param.l = [2,2,1] # Robot links lengths
param.linkMasses = [1,1,1]
param.g = 9.81 # Gravity norm
param.kv = 1 # Joint damping
param.sz = [.2,.3] # Size of objects
param.r = 1e-6 # Control weight term
param.Mu = np.asarray([[2,1,-np.pi/3], [3,2,-np.pi/3]]) # Viapoints 
param.useBoundingBox = True
param.A = np.zeros( (param.nbPoints,2,2) ) # Object orientation matrices

# Main program
# ===============================

# Transformation matrices
for i in range(param.nbPoints):
    orn_t = param.Mu[i,-1]
    param.A[i,:,:] = np.asarray([
        [np.cos(orn_t) , -np.sin(orn_t)],
        [np.sin(orn_t) , np.cos(orn_t)]
    ])

# Precision matrix
Q = np.identity( param.nbVarF  * param.nbPoints)*1e5

# Regularization matrix
R = np.identity( (param.nbData-1) * param.nbVarU ) * param.r

# Time occurence of viapoints
tl = np.linspace(0,param.nbData,param.nbPoints+1)
tl = np.rint(tl[1:]).astype(np.int64)-1
idx = np.array([ i + np.arange(0,param.nbVarX,1) for i in (tl* 2* param.nbVarX)]) 

# iLQR
# ===============================

u = np.zeros(param.nbVarU*(param.nbData-1)) # Initial control command
x0 = np.array([3 * np.pi/4 , -np.pi/2 , - np.pi/4]) # Initial position
v0 = np.array([0, 0, 0]) #initial velocity (in joint space)
x = np.zeros([param.nbData, 2*param.nbVarX])
x[0,:param.nbVarX] = x0
x[0,param.nbVarX:] = v0

for i in range( param.nbIter ):
    # system evolution and Transfer matrix (computed from forward dynamics)
    x, Su0 = forward_dynamics(x, u, param)
    Su = Su0[idx.flatten()]

    f, J = f_reach(x[tl,:param.nbVarX],param)
    du = np.linalg.inv(Su.T @ J.T @ Q @ J @ Su + R) @ (-Su.T @ J.T @ Q @ f.flatten() - u * param.r) # Gauss-Newton update

    # Perform line search
    alpha = 1
    cost0 = f.flatten() @ Q @ f.flatten() + np.linalg.norm(u) * param.r
    
    while True:
        utmp = u + du * alpha
        xtmp, _ = forward_dynamics(x, utmp, param)
        ftmp, _ = f_reach(xtmp[tl,:param.nbVarX],param)
        cost = ftmp.flatten() @ Q @ ftmp.flatten() + np.linalg.norm(utmp) * param.r
        
        if cost < cost0 or alpha < 1e-3:
            u = utmp
            print("Iteration {}, cost: {}, alpha: {}".format(i,cost,alpha))
            break
        alpha /=2
    if abs(cost-cost0)/cost < 1e-3:
        break
        
        
# Plotting
# ===============================
plt.figure()
plt.axis("off")
plt.gca().set_aspect('equal', adjustable='box')

# Get points of interest
f = fkin(x[:,:param.nbVarX],param)
f00 = fkin0(x[0,:param.nbVarX],param)
fT0 = fkin0(x[-1,:param.nbVarX],param)

plt.plot(f00[:,0], f00[:,1], c='black', linewidth=5, alpha=.2)
plt.plot(fT0[:,0], fT0[:,1], c='black', linewidth=5, alpha=.6)

plt.plot(f[:,0], f[:,1], c="black", marker="o", markevery=[0]+tl.tolist()) #,label="Trajectory"

# Plot bounding box or via-points
ax = plt.gca()
color_map = ["deepskyblue","darkorange"]
for i in range(param.nbPoints):
    
    if param.useBoundingBox:
        rect_origin = param.Mu[i,:2] - param.A[i] @ np.array(param.sz)
        rect_orn = param.Mu[i,-1]

        rect = patches.Rectangle(rect_origin,param.sz[0]*2,param.sz[1]*2,np.degrees(rect_orn))
        ax.add_patch(rect)
    else:
        plt.scatter(param.Mu[i,0], param.Mu[i,1], s=100, marker="X", c=color_map[i])

plt.show()
