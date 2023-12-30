"""
Gauss-Newton optimization of an SDF encoded with concatenated cubic polysplines,
by considering unit norm derivatives in the cost function.
In this example, the points used for training are only on the zero-level set (contours of the shape).

Copyright (c) 2023 Idiap Research Institute <https://www.idiap.ch/>
Written by Guillaume Clivaz <guillaume.clivaz@idiap.ch> and Sylvain Calinon <https://calinon.ch>

This file is part of RCFS <https://robotics-codes-from-scratch.github.io/>
License: MIT
"""

import numpy as np
from matplotlib import pyplot as plt


def binomial(n, i):
    if n >= 0 and i >= 0:
        b = np.math.factorial(n) / (np.math.factorial(i) * np.math.factorial(n - i))
    else:
        b = 0
    return b


def block_diag(A, B):
    out = np.zeros((A.shape[0] + B.shape[0], A.shape[1] + B.shape[1]))
    out[: A.shape[0], : A.shape[1]] = A
    out[A.shape[0] :, A.shape[1] :] = B
    return out


def computePsiList(Tmat, param):
    """From a matrix of values (nbIn, N), compute the concatenated basis functions
    (Psi = surface primitive, dPsi = Psi derivatives)to encode the SDF.
    Tmat should have values [t1, t2, ..., tnbIn] in the range [0, 1].

    By example, for 2D:
    1  Tmat is a vector (2, N), [t1, t2] to compute distances.
    1. It determines to which segment t1 and t2 correspond, to select the
    corresponding part of the polynomial matrix.
    3. t residuals are used to fill T = [1, t, t**2, t**3] and dT=[0, 1, 2t, 3t**2]
    4. Psi and dPsi are computed with kronecker product (see 6.3 and 6.4 of RFCS)
    """
    Psi = []
    dPsi1 = []
    dPsi2 = []
    # Compute Psi for each point
    for k in range(0, Tmat.shape[1]):
        T = np.zeros((Tmat.shape[0], param.nbFct))
        dT = np.zeros((Tmat.shape[0], param.nbFct))
        idl = np.zeros((4, 2), dtype="int")

        # Compute Psi for each dimension
        for d in range(0, Tmat.shape[0]):
            # Compute residual within the segment in which the point falls
            tt = np.mod(Tmat[d, k], 1 / param.nbSeg) * param.nbSeg
            # Determine in which segment the point falls in order to evaluate the basis function accordingly
            id = np.round(Tmat[d, k] * param.nbSeg - tt)
            # Handle inputs beyond lower bound
            if id < 0:
                tt = tt + id
                id = 0
            # Handle inputs beyond upper bound
            if id > (param.nbSeg - 1):
                tt = tt + id - (param.nbSeg - 1)
                id = param.nbSeg - 1

            # Evaluate polynomials
            p1 = np.linspace(0, param.nbFct - 1, param.nbFct)
            p2 = np.linspace(0, param.nbFct - 2, param.nbFct - 1)
            T[d, :] = tt**p1
            dT[d, 1:] = p1[1:] @ tt**p2 * param.nbSeg
            idl[:, d] = id.astype("int") * param.nbFct + p1

        # Reconstruct Psi for all dimensions
        Mtmp = np.kron(param.BC[idl[:, 0], :], param.BC[idl[:, 1], :])
        Psi.append(np.kron(T[0, :], T[1, :]) @ Mtmp)
        dPsi1.append(np.kron(dT[0, :], T[1, :]) @ Mtmp)
        dPsi2.append(np.kron(T[0, :], dT[1, :]) @ Mtmp)

    dPsi = np.dstack((np.array(dPsi1), np.array(dPsi2)))
    Psi = np.array(Psi)
    return Psi, dPsi


def computePsiGridFast(t, param):
    """Compute the concatenated basis functions (Psi = surface primitive,
    dPsi = Psi derivatives) to encode the SDF.
    """

    # Time parameters matrix to compute positions
    T0 = np.zeros((t.shape[0], param.nbFct))
    for n in range(param.nbFct):
        T0[:, n] = t**n

    # Matrices for a concatenation of curves
    T = np.kron(np.eye(param.nbSeg), T0)

    # Derivatives
    # Time parameters matrix to compute derivatives
    dT0 = np.zeros((t.shape[0], param.nbFct))
    for n in range(1, param.nbFct):
        dT0[:, n] = n * t ** (n - 1) * param.nbSeg

    # Matrices for a concatenation of curves
    dT = np.kron(np.eye(param.nbSeg), dT0)

    # Transform to multidimensional basis functions
    Psi = np.kron(T, T) @ param.M

    # Transform to multidimensional basis functions
    dPsi = np.kron(dT, T) @ param.M
    dPsi2 = np.kron(T, dT) @ param.M
    dPsi = np.dstack((dPsi, dPsi2))
    return Psi, dPsi


# Residual and Jacobian for tracking objective
def f_track(w, Psi_contour, Mu):
    """Residual and Jacobian for distance tracking objective"""
    f = Psi_contour @ w - Mu
    J = Psi_contour
    return f, J


def f_norm(w, dPsi_eikonal, param, compute_dPsi=True):
    """Residual and Jacobian for unit norm (eikonal) objective"""
    dx = np.hstack((dPsi_eikonal[:, :, 0] @ w, dPsi_eikonal[:, :, 1] @ w))
    f = np.sum(dx**2, axis=1) - 1
    f = f.reshape(-1, 1)
    if not compute_dPsi:
        return f, None

    # Filling dxn (of size N, 2*N) wht diag vector [dx, dy] of size (N, 2)
    N = param.nbDim**param.nbIn
    dxn = np.zeros((param.nbDim**param.nbIn, 2 * param.nbDim**param.nbIn))
    dxn[np.arange(N), np.arange(0, 2 * N, 2)] = dx[:, 0]
    dxn[np.arange(N), np.arange(1, 2 * N, 2)] = dx[:, 1]

    # # Other approach
    # I = np.eye(N)
    # I = np.kron(I, np.ones((1,param.nbIn)))
    # O = np.ones((1,N))
    # dxn = np.multiply(np.kron(O, dx), I)

    J = 2 * dxn @ param.dPsi_reshaped
    return f, J


# General parameters
# ===============================

param = lambda: None  # Lazy way to define an empty class in python

# Concatenated basis functions
param.nbFct = 4  # Number of Bernstein basis functions for each dimension
param.nbSeg = 4  # Number of segments for each dimension
param.nbIn = 2  # Dimension of input data (here: 2D surface embedded in 3D)
param.nbOut = 1  # Dimension of output data (here: height data)
param.nbDim = 40  # Grid size for each dimension (same as sdf01.npy)

# Gauss-Newton optimization
param.nbIter = 100  # Maximum number of iterations
param.qt = 1e0  # Weighting factor for distance tracking
param.qn = 1e-4  # Weighting factor for eikonal objective

# Bézier constraint matrix: B is the polynomial matrix, C the continuity constraints
B0 = np.zeros((param.nbFct, param.nbFct))
for n in range(1, param.nbFct + 1):
    for i in range(1, param.nbFct + 1):
        B0[param.nbFct - i, n - 1] = (
            (-1) ** (param.nbFct - i - n)
            * (-binomial(param.nbFct - 1, i - 1))
            * binomial(param.nbFct - 1 - (i - 1), param.nbFct - 1 - (n - 1) - (i - 1))
        )
B = np.kron(np.eye(param.nbSeg), B0)
C0 = np.array([[1, 0, 0, -1], [0, 1, 1, 2]]).T
C0 = block_diag(np.eye(param.nbFct - 4), C0)
C = np.eye(2)
for n in range(param.nbSeg - 1):
    C = block_diag(C, C0)
C = block_diag(C, np.eye(param.nbFct - 2))
param.BC = B @ C
param.M = np.kron(param.BC, param.BC)


# Load and generate data
# ===============================

# Load distance measurement Xm. Further, only points on contours (SDF level=0)
# will be used.
data = np.load("../data/sdf01.npy", allow_pickle="True").item()
x0 = data["y"].T
dx = data["dx"]  # dx = np.zeros((2, x0.shape[0]))

# Compute Psi and dPsi with basis functions encoding
nb_t = param.nbDim / param.nbSeg  # Number of elements for each dimension of the grid
t = np.linspace(0, 1 - 1 / nb_t, int(nb_t))  # Grid range for each dimension
Psi, dPsi = computePsiGridFast(t, param)

# In this example, all the points sampled are used for the Eikonal objective, but
# a subsample can also be used
Psi_eikonal, dPsi_eikonal = Psi, dPsi

# Superposition weights estimated from least-squares with SDF reference
w_all = np.linalg.pinv(Psi_eikonal) @ x0
x_all = Psi_eikonal @ w_all

dx[0:1, :] = (dPsi_eikonal[:, :, 0] @ w_all).T
dx[1:2, :] = (dPsi_eikonal[:, :, 1] @ w_all).T

# Sample points on the contour of the shape (using pyplot contour)
# to compute distances and derivatives at the desired points
T1, T2 = np.meshgrid(np.linspace(0, 1, param.nbDim), np.linspace(0, 1, param.nbDim))
T1 *= param.nbDim
T2 *= param.nbDim
fig, ax = plt.subplots()
msh = ax.contour(T1, T2, x_all.reshape([param.nbDim, param.nbDim]).T, levels=[0.0])
T_contour = msh.collections[0].get_paths()[0].vertices / param.nbDim
ax.imshow(x0.reshape([param.nbDim, param.nbDim]).T, cmap="gray")
ax.scatter(T_contour[:, 0] * param.nbDim, T_contour[:, 1] * param.nbDim, marker="x")
ax.set_title("Points for the tracking objective sampled on this contour")
plt.show()
Psi_contour, dPsi_contour = computePsiList(T_contour.T, param)
Mu = Psi_contour @ w_all

## Or batch estimation using distances
# x_contour = Psi_contour @ w_all
# wb_contour = np.linalg.pinv(Psi_contour) @ x_contour
# Mu = Psi_contour @ wb_contour


# Solving Gauss-Newton estimation using distances and unit norm derivatives
# =========================================================================

# Initialize weights by loading the SDF of a circle to initialize
data_init = np.load("../data/sdf_circle.npy", allow_pickle="True").item()
x00 = data_init["y"].T
w = np.linalg.pinv(Psi_eikonal) @ x00

# Reformat dPsi for Jacobian of eikonal objective by spliting vertically in two,
# and filling even rows with the first matrix and odd rows with the seconds.
param.dPsi_reshaped = np.empty([2 * param.nbDim**param.nbIn, w.shape[0]])
param.dPsi_reshaped[::2, :] = dPsi_eikonal[:, :, 0]
param.dPsi_reshaped[1::2, :] = dPsi_eikonal[:, :, 1]

for n in range(param.nbIter):
    # Residual and Jacobian for SDF match objective
    ft, Jt = f_track(w, Psi_contour, Mu)

    # Residual and Jacobian for unit norm objective
    fn, Jn = f_norm(w, dPsi_eikonal, param, compute_dPsi=True)

    # Gauss-Newton update (see chapter 6.8 of RCFS.pdf)
    J_inv = np.linalg.pinv(Jt.T @ Jt * param.qt + Jn.T @ Jn * param.qn)
    dw = J_inv @ (-Jt.T @ ft * param.qt - Jn.T @ fn * param.qn)

    # Estimate step size with backtracking line search method
    alpha = 1
    cost0 = ft.T @ ft * param.qt + fn.T @ fn * param.qn
    while True:
        w_tmp = w + dw * alpha
        ft_tmp, _ = f_track(w_tmp, Psi_contour, Mu)
        fn_tmp, _ = f_norm(w_tmp, dPsi_eikonal, param, compute_dPsi=False)
        cost = ft_tmp.T @ ft_tmp * param.qt + fn_tmp.T @ fn_tmp * param.qn
        if cost < cost0 or alpha < 1e-3:
            break
        alpha = alpha * 0.5

    # Gauss-Newton update step
    w = w + dw * alpha
    res = np.linalg.norm(dw * alpha)
    print("Iteration : {} , residual : {}, cost : {}".format(n, res, cost))

    # Stop optimization when solution is reached
    if res < 1e-3:
        break

# Estimated SDF and derivatives
# ===============================

x = Psi @ w
dx[0:1, :] = (dPsi[:, :, 0] @ w).T
dx[1:2, :] = (dPsi[:, :, 1] @ w).T


# Plotting results
# ===============================

x0 *= param.nbDim
x *= param.nbDim
x_all *= param.nbDim

x0 = x0.reshape([param.nbDim, param.nbDim]).T
x = x.reshape((param.nbDim, param.nbDim)).T
dx = dx.reshape((2, param.nbDim, param.nbDim)).T
x_all = x_all.reshape((param.nbDim, param.nbDim)).T


fig = plt.figure(figsize=(20, 10))
ax0 = fig.add_subplot(131)
ax1 = fig.add_subplot(132)
ax2 = fig.add_subplot(133)

ax0.imshow(x0, cmap="gray")
ax0.contour(T1, T2, x0, levels=np.linspace(x0.min(), x0.max(), 25))
ax0.contour(T1, T2, x0, levels=[0.0], linewidths=3.0)
ax0.set_title("Exact SDF")

ax1.imshow(x_all, cmap="gray")
ax1.contour(T1, T2, x_all, levels=np.linspace(x_all.min(), x_all.max(), 25))
ax1.contour(T1, T2, x_all, levels=[0.0], linewidths=3.0)
ax1.set_title("SDF estimated from the reference points (least-square)")

ax2.imshow(x, cmap="gray")
ax2.contour(T1, T2, x, levels=np.linspace(x.min(), x.max(), 25))
ax2.contour(T1, T2, x, levels=[0.0], linewidths=3.0)
ax2.set_title("SDF with Distance + Eikonal objective")

plt.show()
