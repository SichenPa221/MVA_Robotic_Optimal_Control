%% Forward dynamics 
%% 
%% Copyright (c) 2023 Idiap Research Institute <https://www.idiap.ch/>
%% Written by Amirreza Razmjoo <amirreza.razmjoo@idiap.ch> and 
%% Sylvain Calinon <https://calinon.ch>
%% 
%% This file is part of RCFS <https://robotics-codes-from-scratch.github.io/>
%% License: MIT

function FD

%% Parameters
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
param.dt = 1E-2; %Time step size
param.nbData = 500; %Number of datapoints
param.nbPoints = 2; %Number of viapoints
param.nbVarX = 3; %Position space dimension (q1,q2,q3) --> State Space (q1,q2,q3,dq1,dq2,dq3) 
param.nbVarU = 3; %Control space dimension (tau1,tau2,tau3)
param.nbVarF = 3; %Objective function dimension (x1,x2,o)
param.l = [1, 1, 1]; %Robot links lengths
param.m = [1, 1, 1]; %Robot links masses
param.g = 9.81; %gravity norm
param.kv = 1*0; %Joint damping

%% Iterative LQR (iLQR)
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
tau = zeros(param.nbVarU*(param.nbData-1), 1); %Torque commands applied in each joint
x = [-pi/2; 0; 0; zeros(param.nbVarX,1)]; %Initial robot state 

%Auxiliary matrices
l = param.l;
dt = param.dt;
nbDOFs = length(l);
nbData = (size(tau,1) / nbDOFs) + 1;
Lm = triu(ones(nbDOFs)) .* repmat(param.m, nbDOFs, 1);
L = tril(ones(nbDOFs));


%Forward Dynamics
for t=1:nbData-1
%	%Elementwise computation of G, M, C
%	for k=1:nbDOFs
%		G(k,1) = -sum(m(k:nbDOFs)) * param.g * l(k) * cos(q(k,t));
%		for i=1:nbDOFs
%			S = sum(m(k:nbDOFs) .* heaviside([k:nbDOFs]-i+0.1));
%			M(k,i) = l(k) * l(i) * cos(q(k,t) - q(i,t)) * S;
%			C(k,i) = -l(k) * l(i) * sin(q(k,t) - q(i,t)) * S;
%		end
%	end
	%Computation in matrix form of G, M, C
	G = -sum(Lm, 2) .* l' .* cos(L * x(1:nbDOFs,t)) * param.g;
	M = (l' * l) .* cos(L * x(1:nbDOFs,t) - x(1:nbDOFs,t)' * L') .* (Lm.^.5 * Lm.^.5');
	C = -(l' * l) .* sin(L * x(1:nbDOFs,t) - x(1:nbDOFs,t)' * L') .* (Lm.^.5 * Lm.^.5');
	
	G = L' * G;
	C = L' * C;
	M = L' * M * L;
	
	%Update pose
	tau_t = tau((t-1)*nbDOFs+1:t*nbDOFs);
	
	%To debug
	tau_t = zeros(nbDOFs,1);
	G = 0;
	C = 0;
%	M = 1;
	tau_t(1) = 1;
	
	%ddq = M \ (tau + G + C * (q(nbDOFs+1:2*nbDOFs,t)).^2 - inv(L') * q(nbDOFs+1:2*nbDOFs,t) * param.kv) ; %With force damping
	ddx = M \ (tau_t + G + C * (L * x (nbDOFs+1:2*nbDOFs,t)).^2) - L * x(nbDOFs+1:2*nbDOFs,t) * param.kv ; %With joint damping
	x(:,t+1) = x(:,t) + [x(nbDOFs+1:2*nbDOFs,t); ddx] * dt;
end 


%% Plot state space
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
h = figure('position',[10,10,800,800]); hold on; axis off;
h1 = plot(0,0, 'k-','linewidth',2); %link
h2 = plot(0,0, 'k.','markersize',30); %joint and end-effector
axis equal; axis([-1,1,-1,1] * sum(param.l));

% Kinematic simulation
for t=1:param.nbData
	f = fkin0(x(1:param.nbVarX,t), param); %End-effector position
	set(h1, 'XData', f(1,:), 'YData', f(2,:));
	set(h2, 'XData', f(1,:), 'YData', f(2,:));
	drawnow;
	pause(param.dt * 1E-1); %Speeding up the visualization of the simulation 
end

waitfor(h);
end

%%%%%%%%%%%%%%%%%%%%%%
%Forward kinematics for all robot articulations (in robot coordinate system)
function f = fkin0(x, param)
	T = tril(ones(size(x,1)));
	T2 = tril(repmat(param.l, size(x,1), 1));
	f = [T2 * cos(T * x), ...
		 T2 * sin(T * x)]'; 
	f = [zeros(2,1), f];
end


%%%%%%%%%%%%%%%%%%%%%%
% heaviside function (not available in octave)
function y = heaviside(x)
	for i=1:length(x)
		y(i) = 0;
		if x(i) > 0
			y(i) = 1;
		elseif x(i) == 0
			y(i) = 0.5;
		end
	end
end
