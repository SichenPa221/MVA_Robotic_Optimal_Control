%% Inverse kinematics for a planar manipulator
%% 
%% Copyright (c) 2023 Idiap Research Institute <https://www.idiap.ch/>
%% Written by Sylvain Calinon <https://calinon.ch>
%% 
%% This file is part of RCFS <https://robotics-codes-from-scratch.github.io/>
%% License: MIT

function IK

%% Parameters
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
param.dt = 1E-1; %Time step length
param.nbData = 50; %Number of IK iterations
param.nbVarX = 3; %State space dimension (x1,x2,x3)
param.nbVarF = 3; %Objective function dimension (f1,f2,f3, with f3 as orientation)
param.l = [2; 2; 1]; %Robot links lengths

x = ones(param.nbVarX,1) * pi/param.nbVarX; %Initial robot pose
fh = [-3; 1; -pi/2]; %Desired target for the end-effector (position and orientation)
%W = diag([1, 1, 0]); %Precision matrix


%% Inverse kinematics
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
h = figure; hold on; axis off equal;
plot(fh(1,:), fh(2,:), 'r.','markersize',30); %Plot target
for t=1:param.nbData
	f = fkin(x, param); %Forward kinematics (position and orientation of end-effector)
	J = Jkin(x, param); %Jacobian (for end-effector)
	pinvJ = pinv(J); %Pseudoinverse
%	pinvJ = (J' * W * J + eye(param.nbVarX)*1E-8) \ J' * W;	%Damped weighted pseudoinverse
	x = x + pinvJ * (fh - f) * param.dt; %Update state 
	f_rob = fkin0(x, param); %Forward kinematics (for all articulations, including end-effector)
	plot(f_rob(1,:), f_rob(2,:), 'color',ones(1,3)*0.8*(1-t/param.nbData)); %Plot robot
end

waitfor(h);
end


%%%%%%%%%%%%%%%%%%%%%%
%Logarithmic map for R^2 x S^1 manifold
function e = logmap(f, f0)
	e = [f(1:2,:) - f0(1:2,:); ...
	     imag(log(exp(f0(3,:)*1i)' .* exp(f(3,:)*1i).'))'];
end

%%%%%%%%%%%%%%%%%%%%%%
%Forward kinematics for end-effector (in robot coordinate system)
function f = fkin(x, param)
	L = tril(ones(size(x,1)));
	f = [param.l' * cos(L * x); ...
	     param.l' * sin(L * x); ...
	     mod(sum(x,1)+pi, 2*pi) - pi]; %f1,f2,f3, where f3 is the orientation (single Euler angle for planar robot)
end

%%%%%%%%%%%%%%%%%%%%%%
%Forward kinematics for all robot articulations (in robot coordinate system)
function f = fkin0(x, param)
	L = tril(ones(size(x,1)));
	f = [L * diag(param.l) * cos(L * x), ...
	     L * diag(param.l) * sin(L * x)]'; 
	f = [zeros(2,1), f];
end

%%%%%%%%%%%%%%%%%%%%%%
%Jacobian with analytical computation (for single time step)
function J = Jkin(x, param)
	L = tril(ones(size(x,1)));
	J = [-sin(L * x)' * diag(param.l) * L; ...
	      cos(L * x)' * diag(param.l) * L; ...
	      ones(1, size(x,1))]; 
end
