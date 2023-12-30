##    Inverse kinematics computation on a 2D manipulator
##
## Copyright (c) 2023 Idiap Research Institute <https://www.idiap.ch/>
## Written by Sylvain Calinon <https://calinon.ch>
## 
## This file is part of RCFS <https://robotics-codes-from-scratch.github.io/>
## License: MIT

using LinearAlgebra,Plots

T = 50; #Number of datapoints
D = 3; #State space dimension (x1,x2,x3)
l = [2, 2, 1]; #Robot links lengths
fh = [-2; 1]; #Desired target for the end-effector
x = ones(D) * pi/D; #Initial robot pose
L = tril(ones(D,D)); #Transformation matrix

h = scatter(fh[1,:], fh[2,:], markercolor=RGBA(.8,0,0,0), markersize=8, leg=false,  ticks=nothing, border=:none, aspect_ratio=1) #Plot target
for t=1:T
	f = [L * diagm(l) * cos.(L * x)   L * diagm(l) * sin.(L * x)]'; #Forward kinematics (for all articulations, including end-effector)
	J = [-sin.(L * x)' * diagm(l) * L; cos.(L * x)' * diagm(l) * L]; #Jacobian (for end-effector)
	global x += J \ (fh - f[:,end]) * .2; #Update state 
	plot!(h, [0; f[1,:]], [0; f[2,:]], c=:black) #Plot robot
end
display(h) #display final plot result
