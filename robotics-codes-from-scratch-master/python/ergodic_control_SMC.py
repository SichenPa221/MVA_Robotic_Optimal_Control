"""
2D ergodic control formulated as Spectral Multiscale Coverage (SMC) objective,
with a spatial distribution described as a mixture of Gaussians.

Copyright (c) 2023 Idiap Research Institute <https://www.idiap.ch>
Written by Cem Bilaloglu <cem.bilaloglu@idiap.ch> and
Sylvain Calinon <https://calinon.ch>

This file is part of RCFS <https://rcfs.ch>
License: MIT
"""

import numpy as np
#from math import exp
import matplotlib.pyplot as plt


# Helper functions
# ===============================
def hadamard_matrix(n: int) -> np.ndarray:
    """
    Constructs a Hadamard matrix of size n.

    Args:
        n (int): The size of the Hadamard matrix.

    Returns:
        np.ndarray: A Hadamard matrix of size n.
    """
    # Base case: A Hadamard matrix of size 1 is just [[1]].
    if n == 1:
        return np.array([[1]])

    # Recursively construct a Hadamard matrix of size n/2.
    half_size = n // 2
    h_half = hadamard_matrix(half_size)

    # Construct a matrix of ones with size n/2.
    ones_matrix = np.ones((half_size, half_size), dtype=int)

    # Construct a matrix of minus ones with size n/2.
    minus_ones_matrix = -1 * ones_matrix

    # Combine the four sub-matrices to form a Hadamard matrix of size n.
    h = np.empty((n, n), dtype=int)
    for i in range(half_size):
        h[i] = np.concatenate((h_half[i], ones_matrix[i]))
        h[i + half_size] = np.concatenate((h_half[i], minus_ones_matrix[i]))

    return h


# Parameters
# ===============================
nbData = 200  # Number of datapoints
nbFct = 10  # Number of basis functions along x and y
nbVar = 2  # Dimension of datapoints
# Number of Gaussians to represent the spatial distribution
nbGaussian = 2
sp = (nbVar + 1) / 2  # Sobolev norm parameter
dt = 1e-2  # Time step
# Domain limit for each dimension (considered to be 1
# for each dimension in this implementation)
xlim = [0, 1]
L = (xlim[1] - xlim[0]) * 2  # Size of [-xlim(2),xlim(2)]
om = 2 * np.pi / L
u_max = 1e1  # Maximum speed allowed

# Initial point
x0 = [0.1, 0.3]
# this is a regularizer to avoid numerical issues
# when speed is close to zero
u_norm_reg = 1e-1
nbRes = 100

# Desired spatial distribution represented as a mixture of Gaussians (GMM)
# gaussian centers
Mu = np.zeros((nbVar, nbGaussian))
Mu[:, 0] = np.array([0.5, 0.7])
Mu[:, 1] = np.array([0.6, 0.3])
# Gaussian covariances
# Direction vectors for constructing the covariance matrix using
# outer product of a vector with itself then the principal direction
# of covariance matrix becomes the given vector and its orthogonal
# complement
Sigma1_v = [0.3, 0.1]
Sigma2_v = [0.1, 0.2]
# scaling terms
Sigma1_scale = 5e-1
Sigma2_scale = 3e-1
# regularization terms
Sigma1_regularization = np.eye(nbVar) * 5e-3
Sigma2_regularization = np.eye(nbVar) * 1e-2
# cov. matrices
Sigma = np.zeros((nbVar, nbVar, nbGaussian))
# construct the cov. matrix using the outer product
Sigma[:, :, 0] = (
    np.vstack(Sigma1_v) @ np.vstack(Sigma1_v).T * Sigma1_scale + Sigma1_regularization
)
Sigma[:, :, 1] = (
    np.vstack(Sigma2_v) @ np.vstack(Sigma2_v).T * Sigma2_scale + Sigma2_regularization
)
Alpha = np.ones(nbGaussian) / nbGaussian


# Compute Fourier series coefficients w_hat of desired spatial distribution
# ===============================
rg = np.arange(0, nbFct, dtype=float)
KX = np.zeros((nbVar, nbFct, nbFct))
KX[0, :, :], KX[1, :, :] = np.meshgrid(rg, rg)
# Mind the flatten() !!!
# Weighting vector (Eq.(16))
Lambda = np.array(KX[0, :].flatten() ** 2 + KX[1, :].flatten() ** 2 + 1).T ** (-sp)
# Explicit description of w_hat by exploiting the Fourier transform
# properties of Gaussians (optimized version by exploiting symmetries)
op = hadamard_matrix(2 ** (nbVar - 1))
op = np.array(op)
kk = KX.reshape(nbVar, nbFct**2) * om
# compute w_hat
w_hat = np.zeros(nbFct**nbVar)
for j in range(nbGaussian):
    for n in range(op.shape[1]):
        MuTmp = np.diag(op[:, n]) @ Mu[:, j]
        SigmaTmp = np.diag(op[:, n]) @ Sigma[:, :, j] @ np.diag(op[:, n]).T
        cos_term = np.cos(kk.T @ MuTmp)
        exp_term = np.exp(np.diag(-0.5 * kk.T @ SigmaTmp @ kk))
        # Eq.(22) where D=1
        w_hat = w_hat + Alpha[j] * cos_term * exp_term
w_hat = w_hat / (L**nbVar) / (op.shape[1])

# Fourier basis functions (for a discretized map)
# ===============================
xm1d = np.linspace(xlim[0], xlim[1], nbRes)  # Spatial range for 1D
xm = np.zeros((nbGaussian, nbRes, nbRes))  # Spatial range
xm[0, :, :], xm[1, :, :] = np.meshgrid(xm1d, xm1d)
# Mind the flatten() !!!
arg1 = (
    KX[0, :, :].flatten().T[:, np.newaxis] @ xm[0, :, :].flatten()[:, np.newaxis].T * om
)
arg2 = (
    KX[1, :, :].flatten().T[:, np.newaxis] @ xm[1, :, :].flatten()[:, np.newaxis].T * om
)
phim = np.cos(arg1) * np.cos(arg2) * 2 ** (nbVar)  # Fourrier basis functions

# Some weird +1, -1 due to 0 index!!!
xx, yy = np.meshgrid(np.arange(1, nbFct + 1), np.arange(1, nbFct + 1))
hk = np.concatenate(([1], 2 * np.ones(nbFct)))
HK = hk[xx.flatten() - 1] * hk[yy.flatten() - 1]
phim = phim * np.tile(HK, (nbRes**nbVar, 1)).T

# Desired spatial distribution
g = w_hat.T @ phim

# Ergodic control
# ===============================
x = np.array(x0)  # Initial position

wt = np.zeros(nbFct**nbVar)
r_x = np.zeros((nbVar, nbData))
r_g = np.zeros((nbRes**nbVar, nbData))
r_w = np.zeros((nbFct**nbVar, nbData))
r_e = np.zeros((nbData))

for i in range(nbData):
    # Fourier basis functions and derivatives for each dimension
    # (only cosine part on [0,L/2] is computed since the signal
    # is even and real by construction)
    angle = x[:, np.newaxis] * rg * om
    phi1 = np.cos(angle)  # Eq.(18)

    # Gradient of basis functions
    dphi1 = -np.sin(angle) * np.tile(rg, (nbVar, 1)) * om
    phix = phi1[0, xx - 1].flatten()
    phiy = phi1[1, yy - 1].flatten()
    dphix = dphi1[0, xx - 1].flatten()
    dphiy = dphi1[1, yy - 1].flatten()

    dphi = np.vstack([[dphix * phiy], [phix * dphiy]])

    # wt./t are the Fourier series coefficients along trajectory
    # (Eq.(17))
    wt = wt + (phix * phiy).T / (L**nbVar)

    # Controller with constrained velocity norm
    u = -dphi @ (Lambda * (wt / (i + 1) - w_hat))  # Eq.(24)
    u = u * u_max / (np.linalg.norm(u) + u_norm_reg)  # Velocity command

    x = x + (u * dt)  # Update of position
    # Log data
    r_x[:, i] = x
    # Reconstructed spatial distribution (for visualization)
    r_g[:, i] = (wt / (i + 1)).T @ phim
    # Fourier coefficients along trajectory (for visualization)
    r_w[:, i] = wt / (i + 1)
    # Reconstruction error evaluation
    r_e[i] = np.sum((wt / (i + 1) - w_hat) ** 2 * Lambda)

# Plot
# ===============================
fig, ax = plt.subplots(1, 3, figsize=(16, 8))
G = np.reshape(g, [nbRes, nbRes])  # original distribution
G = np.where(G > 0, G, 0)

# G = np.reshape(r_g[:, -1], [nbRes, nbRes])  # reconstructed spatial distribution

# x
X = np.squeeze(xm[0, :, :])
Y = np.squeeze(xm[1, :, :])
ax[0].contourf(X, Y, G, cmap="gray_r")
ax[0].plot(r_x[0, :], r_x[1, :], linestyle="-", color="black")
ax[0].plot(r_x[0, 0], r_x[1, 0], marker=".", color="black", markersize=10)
ax[0].set_aspect("equal", "box")

# w
ax[1].set_title(r"$w$")
w = ax[1].imshow(np.reshape(wt / nbData, [nbFct, nbFct]).T, cmap="gray_r")

# w_hat
ax[2].set_title(r"$\hat{w}$")
ax[2].imshow(np.reshape(w_hat, [nbFct, nbFct]).T, cmap="gray_r")
# plt.savefig('ergodic_weight.pdf')
plt.show()
