%% Inverse kinematics with nullspace projection operator and line search
%% (position and orientation tracking as primary or secondary tasks)
%% 
%% Copyright (c) 2023 Idiap Research Institute <https://www.idiap.ch/>
%% Written by Teguh Lembono <teguh.lembono@idiap.ch> and 
%% Sylvain Calinon <https://calinon.ch>
%% 
%% This file is part of RCFS <https://robotics-codes-from-scratch.github.io/>
%% License: MIT

function IK_nullspace

%% Parameters
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%param.dt = 1E-2; %Time step
param.nbIter = 100; %Number of iterations
param.Kp = 1E0; %Position tracking gain
param.Ko = 1E0; %Orientation tracking gain
param.fh = [4; 1; -pi/2]; %Target

param.mu_p = 1E-4; %regularization for position
param.mu_o = 1E-4; %regularization for orientation
param.alpha_fac = 0.5;

%% Robot parameters 
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
param.nbVarX = 3; %Number of articulations
param.l = [2; 2; 1]; %Link lengths

%% Initialization
x = [3*pi/4; -pi/2; -pi/4]; %Initial pose
h = figure('position',[10,10,800,800],'color',[1,1,1]); hold on; axis off;
plot(param.fh(1,:), param.fh(2,:), 'r.','markersize',30); %Plot target


%% IK with nullspace
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
for t=1:param.nbIter
	%Calculate forward kinematics	
	f = fkin(x, param);

	%Jacobians and pseudoinverses
	J = Jkin(x, param);
	Jp = J(1:2,:); %Jacobian for position 
	Jo = J(3,:); %Jacobian for orientation
	pinvJp = (Jp' * Jp + eye(param.nbVarX) * param.mu_p) \ Jp'; %Damped pseudoinverse
	Np = eye(param.nbVarX) - pinvJp * Jp; %Nullspace projection operators

	%Error corrections
	dfp = param.Kp * (param.fh(1:2) - f(1:2));
	dfo = param.Ko * (param.fh(3) - f(3));

	%Prioritized tracking task 
	%(position tracking as primary task, orientation tracking as secondary task)

	%1. Determine the first step size
	alpha1 = 1;
	cost10 = (param.fh(1:2) - f(1:2))' * (param.fh(1:2) - f(1:2)); %current cost
	cost1 = 5 * cost10; %initialize with something big
	dxp = pinvJp * dfp; %search direction for first cost
	while cost1 > cost10 + 1e-4 %1e-4 is the tolerance
		dxp_try = alpha1 * dxp;
		x_next = x + dxp_try;
		f = fkin(x_next, param);
		cost1 = (param.fh(1:2) - f(1:2))' * (param.fh(1:2) - f(1:2));
		alpha1 = alpha1 * param.alpha_fac;
	end
	dxp = dxp_try;
	
	x_next = x + dxp;
	f = fkin(x_next, param);
  
	
	%2. Determine the second step size
	alpha2 = 1;
	cost10 = (param.fh(1:2) - f(1:2))' * (param.fh(1:2) - f(1:2)); %current cost1
	cost20 = (param.fh(3) - f(3))' * (param.fh(3) - f(3)); %current cost2

	cost1 = 5 * cost10 + 1e1; %initialize with something big
	cost2 = 5 * cost20; %initialize with something big

	JoNp = Jo * Np;
	pinvJoNp = JoNp' / (JoNp * JoNp' + eye(1) * param.mu_o);
	dxo = Np * pinvJoNp * (dfo - Jo * dxp); %search direction for second cost
  
	%Decrease alpha2 until the cost 1 and cost2 are lower or the same
	%(with some tolerance)
	while (cost1 > cost10 + 1e-8) || (cost2 > cost20 + 1e-4) %1e-8 is the tolerance
		dxo_try = alpha2 * dxo;
		x_next = x + dxp + dxo_try;
		f = fkin(x_next, param);
		cost1 = (param.fh(1:2) - f(1:2))' * (param.fh(1:2) - f(1:2));
		cost2 = (param.fh(3) - f(3))' * (param.fh(3) - f(3));
		alpha2 = alpha2 * param.alpha_fac;
	end
	dxo = dxo_try;
	
	dx = dxp + dxo;
	x = x + dx ; %Update pose

	fprintf('Cost_p: %f, Cost_o: %f, alpha1: %f, alpha2:%f\n', ...
	(param.fh(1:2) - f(1:2))' * (param.fh(1:2) - f(1:2)), ...
	(param.fh(3) - f(3)) * (param.fh(3) - f(3)), alpha1, alpha2);
	
	%Plot the robot
	fs = fkin0(x, param);
	%plot(ftmp(1,:), ftmp(2,:), '-','linewidth',4,'color',[.8 .8 .8]);
	plot(fs(1,:), fs(2,:), '-','linewidth',4,'color',ones(1,3)*(1-t/param.nbIter)); 

end
%Plot the final configuration
fs = fkin0(x, param);
plot(fs(1,:), fs(2,:), '-','linewidth',4,'color',[.4 .4 .4]); 
axis equal; 
waitfor(h);
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
%Forward kinematics for end-effector 
function f = fkin(x, param)
	L = tril(ones(size(x,1)));
	f = [param.l' * cos(L * x); ...
	     param.l' * sin(L * x); ...
	     mod(sum(x,1)+pi, 2*pi) - pi]; %f1,f2,f3, where f3 is the orientation (single Euler angle for planar robot)
end

%%%%%%%%%%%%%%%%%%%%%%
%Jacobian with numerical computation
function J = Jkin(x, l)
	e = 1E-6;
	D = size(x,1);
	
%	%Slow for-loop computation
%	J = zeros(3,D);
%	for n=1:D
%		xtmp = x;
%		xtmp(n) = xtmp(n) + e;
%		ftmp = fkin(xtmp, l); %Forward kinematics 
%		f = fkin(x, l); %Forward kinematics 
%		J(:,n) = (ftmp - f) / e; %Jacobian for end-effector
%	end
	
	%Fast matrix computation
	X = repmat(x, [1, D]);
	F1 = fkin(X, l);
	F2 = fkin(X + eye(D) * e, l);
	J = (F2 - F1) / e;
end
