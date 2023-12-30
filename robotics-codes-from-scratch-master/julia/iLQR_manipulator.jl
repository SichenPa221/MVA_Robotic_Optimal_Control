## iLQR applied to a planar manipulator for a viapoints task (batch formulation)
##
## Copyright (c) 2023 Idiap Research Institute <https://www.idiap.ch/>
## Written by Tobias Löw <tobias.loew@idiap.ch> and
## Sylvain Calinon <https://calinon.ch>
## 
## This file is part of RCFS <https://robotics-codes-from-scratch.github.io/>
## License: MIT

using LinearAlgebra #,Plots
using Parameters

@with_kw struct Params
	dt::Float64 = 1e-2 # Time step length
	nbData::Int64 = 50 # Number of datapoints
	nbIter::Int64 = 100 # Maximum number of iterations for iLQR
	nbPoints::Int64 = 2 # Number of viapoints
	nbVarX::Int64 = 3 # State space dimension (x1,x2,x3)
	nbVarU::Int64 = 3 # Control space dimension (dx1,dx2,dx3)
	nbVarF::Int64 = 3 # Objective function dimension (f1,f2,f3, with f3 as orientation)
	l::Array{Float64} = [2,2,1] # Robot links lengths
	sz::Array{Float64} = [.2,.3] # Size of objects
	r::Float64 = 1e-6 # Control weight term
	mu::Array{Float64} = transpose([2 1 -pi/6; 3 2 -pi/3]) # Viapoints 
	A::Array{Float64} = zeros(2,2,nbPoints) # Object orientation matrices
	useBoundingBox::Bool = true # Consider bounding boxes for reaching cost
end

# Logarithmic map for R^2 x S^1 manifold
function logmap(f::Array{Float64},f0::Array{Float64})
	position_error = f[1:2,:] - f0[1:2,:]
	orientation_error = Array{Float64}(conj.(imag.(log.(transpose(conj.(exp.(f0[3,:]*1im))).*transpose(exp.(f[3,:]*1im))))))

	return [position_error; orientation_error]
end

# Forward kinematics for end-effector (in robot coordinate system)
function fkin(x::Array{Float64}, param::Params)
	L::Array{Float64} = LowerTriangular(ones(param.nbVarX, param.nbVarX))

	f::Array{Float64} = [
			param.l'cos.(L * x);
			param.l'sin.(L * x);
			mod.(sum(x,dims=1).+pi, 2*pi) .- pi
		] # f1,f2,f3, where f3 is the orientation (single Euler angle for planar robot)

	return f
end

# Forward Kinematics for all joints
function fkin0(x::Array{Float64},param::Params)
	L::Array{Float64} = LowerTriangular(ones(param.nbVarX, param.nbVarX))
	
	f::Array{Float64} = hcat(
			L * Diagonal(param.l) * cos.(L * x),
			L * Diagonal(param.l) * sin.(L * x)
		)

	return [zeros(1,2); f]
end

function Jkin(x::Array{Float64},param::Params)
	L::Array{Float64} = LowerTriangular(ones(param.nbVarX, param.nbVarX))

	J::Array{Float64} = [
		-transpose(sin.(L * x)) * Diagonal(param.l) * L;
		transpose(cos.(L * x)) * Diagonal(param.l) * L;
		ones(1,param.nbVarX)
	]

	return J
end

# Cost and gradient
function f_reach(x::Array{Float64},param::Params)
	f = logmap(fkin(x,param), param.mu)
	J = zeros(size(x)[2] * param.nbVarF, size(x)[2] * param.nbVarX)

	for t in range(1,size(x)[2])
		f[1:2,t] = param.A[:,:,t] * f[1:2,t] # Object oriented forward kinematics

		Jtmp = Jkin(x[:,t],param)
		Jtmp[1:2,:] = param.A[:,:,t] * Jtmp[1:2,:] # Object centered Jacobian
		
		if param.useBoundingBox
			for i in range(1,2)
				if abs(f[i,t]) < param.sz[i]
					f[i,t] = 0
					Jtmp[i] = 0
				else
					f[i,t] -= sign(f[i,t]) * param.sz[i]
				end 
			end 
		end 
	
		J[(t-1)*param.nbVarF+1:t*param.nbVarF, (t-1)*param.nbVarX+1:t*param.nbVarX] = Jtmp
	end 

	return f,J
end

param = Params()

# Object rotation matrices
for i in range(1, param.nbPoints)
	orn_t = param.mu[3,i]
	param.A[:,:,i] = Array{Float64}([cos(orn_t), -sin(orn_t), sin(orn_t), cos(orn_t)])
end

# Precision matrix
Q = I #Matrix{Float64}(I,param.nbVarF * param.nbPoints,param.nbVarF * param.nbPoints)

# Control weight matrix
R = param.r * I #identity((param.nbData-1) * param.nbVarU) * param.r

# Time occurrence of viapoints
tl = (Array{Int64}(range(0, stop=param.nbData, length=param.nbPoints+1)) .- 1)[2:end] .+ 1
idx = Array{Int64}(zeros(param.nbVarX,param.nbVarX-1))
for i in range(1,param.nbVarX-1)
	idx[:,i] = Array{Int64}(((tl[i]-1)*param.nbVarX) .+ range(1,param.nbVarX)) #.+ 1
end

# Transfer matrices (for linear system as single integrator)
Su0 = [zeros(param.nbVarX, param.nbVarX*(param.nbData-1));
      LowerTriangular(kron(ones(param.nbData-1, param.nbData-1), Matrix{Float64}(I,param.nbVarX,param.nbVarX) * param.dt))] 
Sx0 = Array{Float64}(kron(ones(param.nbData), Matrix{Float64}(I,param.nbVarX,param.nbVarX)))
Su = Su0[idx[:],:] # We remove the lines that are out of interest

# iLQR
# ===============================

x = zeros(param.nbVarX * (param.nbData)) # Initial control command
u = zeros(param.nbVarU * (param.nbData-1)) # Initial control command
x0 = [3*pi/4; -pi/2; -pi/4] # Initial state

for i in range(1,param.nbIter)
	global x = Su0 * u + Sx0 * x0 # System evolution
	global x = reshape(transpose(x),(param.nbVarX,param.nbData))

	f,J = f_reach(x[:,tl],param)
	
	du = inv(transpose(Su) * transpose(J) * Q * J * Su + R) * (-transpose(Su) * transpose(J) * Q * f[:] - u * param.r) # Gradient

	# Estimate step size with backtracking line search method
	alpha = 1.0
	cost0 = transpose(f[:]) * Q * f[:] + norm(u) * param.r

	while true
		utmp = u + du * alpha
		xtmp = Su0 * utmp + Sx0 * x0
		xtmp = reshape(transpose(xtmp),(param.nbVarX,param.nbData))
		
		ftmp,Jtmp = f_reach(xtmp[:,tl],param)

		cost = transpose(ftmp[:]) * Q * ftmp[:] + norm(utmp) * param.r
		if cost < cost0 || alpha < 1e-3
			global u = utmp
			print("Iteration ")
			print(i)
			print(", cost: ")
			print(cost)
			print(", alpha: ")
			println(alpha)
			break
		end
		alpha /= 2.0
	end
	
	if norm(du * alpha) < 1e-2
		# println(i)
		break # Stop iLQR iterations when solution is reached
	end
end
