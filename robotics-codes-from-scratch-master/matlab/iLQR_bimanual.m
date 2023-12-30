%% iLQR applied to a planar bimanual robot for a tracking problem involving 
%% the center of mass (CoM) and the end-effector (batch formulation)
%% 
%% Copyright (c) 2023 Idiap Research Institute <https://www.idiap.ch/>
%% Written by Sylvain Calinon <https://calinon.ch>
%% 
%% This file is part of RCFS <https://robotics-codes-from-scratch.github.io/>
%% License: MIT

function iLQR_bimanual

%% Parameters
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
param.dt = 1E0; %Time step size
param.nbIter = 100; %Maximum number of iterations for iLQR
param.nbPoints = 1; %Number of viapoints
param.nbData = 30; %Number of datapoints
param.nbVarX = 5; %State space dimension ([q1,q2,q3] for left arm, [q1,q4,q5] for right arm)
param.nbVarU = param.nbVarX; %Control space dimension (dq1,dq2,dq3,dq4,dq5)
param.nbVarF = 4; %Task space dimension ([x1,x2] for left end-effector, [x3,x4] for right end-effector)
param.l = ones(param.nbVarX,1) * 2; %Robot links lengths
param.r = 1E-5; %Control weighting term
param.Mu = [-1; -1.5; 4; 2]; %Target point for end-effectors
%param.Mu = [-3; 1.5; 4.5; 3]; %Target point for end-effectors
param.MuCoM = [0; 1.4]; %Target point for center of mass

R = speye(param.nbVarU * (param.nbData-1)) * param.r; %Control weight matrix (at trajectory level)
Q = kron(speye(param.nbPoints), diag([1, 1, 0, 0])); %Precision matrix for end-effectors tracking 
Qc = kron(speye(param.nbData), diag([1, 1])); %Precision matrix for continuous CoM tracking 

%Time occurrence of viapoints
tl = linspace(1, param.nbData, param.nbPoints+1);
tl = round(tl(2:end));
idx = (tl - 1) * param.nbVarX + [1:param.nbVarX]';


%% Iterative LQR (iLQR)
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
u = zeros(param.nbVarU*(param.nbData-1), 1);
x0 = [pi/2; pi/2; pi/3; -pi/2; -pi/3]; %Initial pose

%Transfer matrices (for linear system as single integrator)
Su0 = [zeros(param.nbVarX, param.nbVarX*(param.nbData-1)); tril(kron(ones(param.nbData-1), eye(param.nbVarX)*param.dt))];
Sx0 = kron(ones(param.nbData,1), eye(param.nbVarX));
Su = Su0(idx,:);

for n=1:param.nbIter	
	x = reshape(Su0 * u + Sx0 * x0, param.nbVarX, param.nbData); %System evolution
	[f, J] = f_reach(x(:,tl), param); %Forward kinematics and Jacobian for end-effectors
	[fc, Jc] = f_reach_CoM(x, param); %Forward kinematics and Jacobian for center of mass	
	du = (Su' * J' * Q * J * Su + Su0' * Jc' * Qc * Jc * Su0 + R) \ (-Su' * J' * Q * f - Su0' * Jc' * Qc * fc - u * param.r); %Gradient
	
	%Estimate step size with line search method
	alpha = 1;
	cost0 = f' * Q * f + fc' * Qc * fc + norm(u)^2 * param.r; %for end-effectors and CoM
	while 1
		utmp = u + du * alpha;
		xtmp = reshape(Su0 * utmp + Sx0 * x0, param.nbVarX, param.nbData);
		ftmp = f_reach(xtmp(:,tl), param);
		fctmp = f_reach_CoM(xtmp, param);
		cost = ftmp' * Q * ftmp + fctmp' * Qc * fctmp + norm(utmp)^2 * param.r; %for end-effectors and CoM
		if cost < cost0 || alpha < 1E-3
			break;
		end
		alpha = alpha * 0.5;
	end
	u = u + du * alpha; 
	
	if norm(du * alpha) < 1E-2
		break; %Stop iLQR when solution is reached
	end
end
disp(['iLQR converged in ' num2str(n) ' iterations.']);


%% Plot state space
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
tl = [1, tl];
colMat = lines(param.nbPoints*2);

h = figure('position',[10,10,1200,800],'color',[1,1,1]); hold on; axis off;
%Plot bimanual robot
ftmp = fkin0(x(:,1), param);
plot(ftmp(1,:), ftmp(2,:), '-','linewidth',4,'color',[.8 .8 .8]);
ftmp = fkin0(x(:,tl(end)), param);
plot(ftmp(1,:), ftmp(2,:), '-','linewidth',4,'color',[.4 .4 .4]);
%Plot CoM
fc = fkin_CoM(x, param); %Forward kinematics for center of mass 
plot(fc(1,1), fc(2,1), 'o','linewidth',4,'markersize',12,'color',[.5 .5 .5]); %Plot CoM
plot(fc(1,tl(end)), fc(2,tl(end)), 'o','linewidth',4,'markersize',12,'color',[.2 .2 .2]); %Plot CoM
%Plot end-effectors targets
for t=1:param.nbPoints
	plot(param.Mu(1,t), param.Mu(2,t), '.','markersize',50,'color',colMat((t-1)*2+2,:));
	%plot(param.Mu(3,t), param.Mu(4,t), '.','markersize',50,'color',colMat((t-1)*2+1,:));
end
%Plot CoM target
plot(param.MuCoM(1,:), param.MuCoM(2,:), 'ro','linewidth',4,'markersize',12); %Plot end-effector target
%Plot end-effectors paths
ftmp = fkin(x, param);
plot(ftmp(1,:), ftmp(2,:), 'k-','linewidth',1);
plot(ftmp(3,:), ftmp(4,:), 'k-','linewidth',1);
plot(ftmp(1,tl), ftmp(2,tl), 'k.','markersize',20);
plot(ftmp(3,tl), ftmp(4,tl), 'k.','markersize',20);
axis equal; 

waitfor(h);
end 


%%%%%%%%%%%%%%%%%%%%%%
%Forward kinematics for all articulations of a bimanual robot (in robot coordinate system)
function f = fkin0(x, param)
	L = tril(ones(3));
	fl = [L * diag(param.l(1:3)) * cos(L * x(1:3)), ...
	      L * diag(param.l(1:3)) * sin(L * x(1:3))]'; 
	fr = [L * diag(param.l([1,4:5])) * cos(L * x([1,4:5])), ...
	      L * diag(param.l([1,4:5])) * sin(L * x([1,4:5]))]';
	f = [fliplr(fl), zeros(2,1), fr];
end

%%%%%%%%%%%%%%%%%%%%%%
%Forward kinematics for bimanual robot (in robot coordinate system)
function f = fkin(x, param)
	L = tril(ones(3));
	f = [param.l(1:3)' * cos(L * x(1:3,:)); ...
	     param.l(1:3)' * sin(L * x(1:3,:)); ...
	     param.l([1,4:5])' * cos(L * x([1,4:5],:)); ...
	     param.l([1,4:5])' * sin(L * x([1,4:5],:))];
end

%%%%%%%%%%%%%%%%%%%%%%
%Forward kinematics for center of mass of a bimanual robot (in robot coordinate system)
function f = fkin_CoM(x, param)
	L = tril(ones(3));
	f = [param.l(1:3)' * L * cos(L * x(1:3,:)) + param.l([1,4:5])' * L * cos(L * x([1,4:5],:)); ...
	     param.l(1:3)' * L * sin(L * x(1:3,:)) + param.l([1,4:5])' * L * sin(L * x([1,4:5],:))] / 6;
end

%%%%%%%%%%%%%%%%%%%%%%
%Jacobian for bimanual robot (in robot coordinate system)
function J = Jkin(x, param)
	L = tril(ones(3));
	J = [-sin(L * x([1,2,3]))' * diag(param.l([1,2,3])) * L; ...
	      cos(L * x([1,2,3]))' * diag(param.l([1,2,3])) * L];
	J(3:4,[1,4:5]) = [-sin(L * x([1,4,5]))' * diag(param.l([1,4,5])) * L; ...
	                   cos(L * x([1,4,5]))' * diag(param.l([1,4,5])) * L];
end

%%%%%%%%%%%%%%%%%%%%%%
%Jacobian for center of mass of a bimanual robot (in robot coordinate system)
function J = Jkin_CoM(x, param)
	L = tril(ones(3));
	Jl = [-sin(L * x(1:3))' * L * diag(param.l(1:3)' * L); ...
	       cos(L * x(1:3))' * L * diag(param.l(1:3)' * L)] / 6;
	Jr = [-sin(L * x([1,4:5]))' * L * diag(param.l([1,4:5])' * L); ...
	       cos(L * x([1,4:5]))' * L * diag(param.l([1,4:5])' * L)] / 6;
	J = [(Jl(:,1) + Jr(:,1)), Jl(:,2:end), Jr(:,2:end)]; %Jacobian for center of mass
end

%%%%%%%%%%%%%%%%%%%%%%
%Cost and gradient for end-effectors
function [f, J] = f_reach(x, param)
	f = fkin(x, param) - param.Mu;
	f = f(:);
	
	J = []; 
	for t=1:size(x,2)
		J = blkdiag(J, Jkin(x(:,t), param));
	end
end

%%%%%%%%%%%%%%%%%%%%%%
%Cost and gradient for center of mass of bimanual robot
function [f, J] = f_reach_CoM(x, param)
	f = fkin_CoM(x, param) - param.MuCoM; 
	f = f(:);

	J = []; 
	for t=1:size(x,2)
		J = blkdiag(J, Jkin_CoM(x(:,t), param));
	end
end
