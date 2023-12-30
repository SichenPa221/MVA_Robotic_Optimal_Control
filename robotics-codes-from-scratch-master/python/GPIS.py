'''
Gaussian process implicit surface (GPIS) representation 

Copyright (c) 2023 Idiap Research Institute <https://www.idiap.ch/>
Written by Cem Bilaloglu <cem.bilaloglu@idiap.ch> and
Sylvain Calinon <https://calinon.ch>

This file is part of RCFS <https://robotics-codes-from-scratch.github.io/>
License: MIT
'''

import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial.distance import cdist

# Utility Functions
######################################################################

# Not required for minimal example since we don't use dK
def substr(x1,x2):
    """Broadcasts arrays with different shape
    by introducing an additional axis, then perform the substraction. 
    Default behaviour with '-' in Matlab, externally added in numpy.

    Parameters
    ----------
    x1 : numpy arr.
    x2 : numpy arr.

    Returns
    -------
    sub_arr : numpy arr.
    """
    dim1 = x1.shape[0]
    dim2 = x2.shape[0]
    sub_arr = x1[:,np.newaxis]-x2
    return sub_arr

def covFct(x1,x2,p):
    """Computes covariance matrix K(x1,x2) with the covariance kernel 
    function k(x_i,x_j). Where x_i=x1[i] and x_j=x2[j]. Kernel fct. 
    provides covariance bw/ any two sample locations x1[i], x2[j] and 
    it is chosen depending on the desired process behavior.
    
    Implemented kernel covariance fcts.:
    RBF:radial basis function or squared exponential kernel covariance 
    is useful for homogenous process i.e. depends only on euclidian 
    distance btw. x1-x2 and not the direction

    Parameters
    ----------
    x1 : numpy arr.
    x2 : numpy arr.

    Returns
    -------
    K : numpy arr.
        Positive semi-definite covariance matrix, whose elements are
        k(x_i,x_j) corresponding to covariance kernel fct.
    dK : numpy arr.
        Derivative of the covariance mtx. (not required for the 
        minimal example)
    """
    squared_euclidian_dist = cdist(x1.T, x2.T)**2
    K = p[0] * np.exp(-p[1] * squared_euclidian_dist) # kernel mtx.

    # !Derivatives are not required for the basic implementation
    dK = np.zeros((np.shape(K)[0],np.shape(K)[1],2))
    # derivatives along x1
    dK[:,:,0] = -p[1] * p[0] * np.exp(-p[0] * 
                squared_euclidian_dist) * substr(x1[0,:],x2[0,:])
    # derivatives along x2
    dK[:,:,1] = -p[1] * p[0] * np.exp(-p[0] *
                squared_euclidian_dist) * substr(x1[1,:],x2[1,:])
    return K,dK


# Generate Data
######################################################################
"""
# Interchangeable with the code block on below just add/remove '#' on above line
# Run this block for random sample pts. 
# Note that random pts usually result in weird concave shapes
nbDataPoints = 5
x1 = np.random.randn(nbDataPoints)
x2 = np.random.randn(nbDataPoints)
y = np.random.choice(np.array([-1,0,1]),nbDataPoints)
#"""

#"""
# Interchangeable with the code block on the top just add/remove '#' on above line

# Run this block for the sample pts. given below 
x1 = [0.2, 0.4, 0.6, -0.4, 0.6, 0.9]
x2 = [0.5, 0.5, 0.5,  0.8, 0.1, 0.6]
# 1: interior point, 0: boundary point, -1: exterior point
y = [ 0,   0,   1,   -1,  -1,  -1]
#"""

# observed data points x is 2-D input, y is output
x = np.vstack((x1,x2))
y = np.array(y)

# Parameters
######################################################################
# hyperparameters p corresponding I/O scales of the problem
p = [1e-2, 5e-2**-1, 1e-4]

# (Optional) adapts the grid size based on the data points
######################################################################
delta_x = 1
x1_min,x2_min = np.min(x[0,:]) - delta_x, np.min(x[1,:]) - delta_x
x1_max,x2_max = np.max(x[0,:]) + delta_x, np.max(x[1,:]) + delta_x
print(f"x1 min:{x1_min},x1 max:{x1_max}"
        f" | x2 min:{x2_min},x2 max:{x2_max}")

## Reproduction with GPR
######################################################################
nbDataRepro = 100 # number of datapoints per axis to reproduce
                   # for visualization

# meshgrid returns in the reverse order as matlab?
X1s, X2s = np.meshgrid(
                        np.linspace(x1_min,x1_max,nbDataRepro),
                        np.linspace(x1_min,x1_max,nbDataRepro))
# I dont get why I should place first X2s 
# for [X1s(:)';X2s(:)'] of Matlab
xs = np.vstack((X2s.flatten(), X1s.flatten()))

K,_ = covFct(x, x, p) # K(x,x)
K = K + p[2] * np.eye(x.shape[1]) # add noise to observations in y
Ks,_ = covFct(xs, x, p) # K(x*,x)

# check dims. of matrices
print(f"Shapes of \nK(x^*,x):{Ks.shape}\nK(x,x):"
    f"{K.shape}\ny:{y.shape}")

# assume mu(x) and mu(x*) are both 0
ys = Ks@np.linalg.inv(K)@y.T # conditinal gaussian Eqn.1

# Plots
######################################################################
fig, ax = plt.subplots()
ax.set(aspect=1, xlim=[x1_min, x1_max], ylim=[x2_min, x2_max])
ax.axis('off')
for i in range(len(y)): # go over the data points
    if y[i]==-1:
        # exterior point
        color_list = [0, .6, 0] # green
    elif y[i]==0:
        # boundary point
        color_list = [.8,.4,0] # orange
    elif y[i]==1:
        # interior point
        color_list = [.8,0,0] # red
    # plot the datapoint by its specified color
    ax.plot(x[0,i],x[1,i], marker='.',markersize=8,color=color_list)
# heatmap shows the precision (inverse of uncertainity)
ax.pcolor(X2s, X1s,
            np.reshape(-1*ys,(nbDataRepro,nbDataRepro)),
            cmap='RdYlGn', vmin=-1, vmax=1)# cmap Red-Yellow-Green
# contour shows the estimated object boundary
ax.contour(X2s, X1s, 
            np.reshape(ys,(nbDataRepro,nbDataRepro)),
            colors='orange',levels=1) # levels determines number of 
            #contours           
plt.show()
# plt.savefig('GPIS.pdf')
