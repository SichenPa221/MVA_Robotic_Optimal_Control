%% Batch iLQR with computation of the manipulator dynamics
%% 
%% Copyright (c) 2023 Idiap Research Institute <https://www.idiap.ch/>
%% Written by Amirreza Razmjoo <amirreza.razmjoo@idiap.ch> and 
%% Sylvain Calinon <https://calinon.ch>
%% 
%% This file is part of RCFS <https://robotics-codes-from-scratch.github.io/>
%% License: MIT

function iLQR_manipulator_dynamics


%% Parameters
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
param.dt = 1E-2; %Time step size
param.nbData = 50; %Number of datapoints
param.nbIter = 300; %Number of iterations for iLQR
param.nbPoints = 2; %Number of viapoints
param.nbVarX = 3; %Position space dimension (q1,q2,q3) --> State Space (q1,q2,q3,dq1,dq2,dq3) 
param.nbVarU = 3; %Control space dimension (tau1,tau2,tau3)
param.nbVarF = 3; %Objective function dimension (x1,x2,o)
param.l = [2, 2, 1]; %Robot links lengths
param.m = [2, 3, 4]; %Robot links masses
param.sz = [.2, .3]; %Size of objects
param.r = 1E-6; %Control weight term

param.Mu = [[2; 1; -pi/3], [3; 2; -pi/3]]; %Viapoints 
for t=1:param.nbPoints
	param.A(:,:,t) = [cos(param.Mu(3,t)), -sin(param.Mu(3,t)); sin(param.Mu(3,t)), cos(param.Mu(3,t))]; %Orientation
end

R = speye((param.nbData-1)*param.nbVarU) * param.r; %Control weight matrix (at trajectory level)
Q = speye(param.nbVarF * param.nbPoints) * 1E5; %Precision matrix
% Q = kron(eye(param.nbPoints), diag([1E0, 1E0, 0])); %Precision matrix (by removing orientation constraint)

%Time occurrence of viapoints
tl = linspace(1, param.nbData, param.nbPoints+1);
tl = round(tl(2:end));
idx = (tl - 1) * 2 * param.nbVarX + [1:param.nbVarX]';


%% Iterative LQR (iLQR)
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
tau = zeros(param.nbVarU*(param.nbData-1), 1); %Initial commands (Torque commands) 
x0 = [3 * pi/4; -pi/2; -pi/4; zeros(param.nbVarX,1)]; %Initial robot pose (Relative angles and velocities)

%Conversion of the state variables to relative form
L = tril(ones(param.nbVarX));
x_abs0 = blkdiag(L,L) * x0; %Absolute angles and velocities

for n=1:param.nbIter
	
	u = kron(eye(param.nbData-1),inv(L')) * tau; %Generalized forces
	%System evolution and Transfer matrix (Computed from forward dynamics)
	[x_abs, Su0] = fwdDyn(x_abs0, u, param);
	x = inv(blkdiag(L,L)) * x_abs; %Relative angles
	% Conversion of the variables
	Su0 = inv(kron(eye(param.nbData),blkdiag(L,L))) * Su0 * kron(eye(param.nbData-1),inv(L')); 
	Su = Su0(idx,:);
	[f, J] = f_reach(x(1:param.nbVarX,tl), param);
	dtau = (Su' * J' * Q * J * Su + R) \ (-Su' * J' * Q * f(:) - tau * param.r); %Gradient
	
	%Estimate step size with backtracking line search method
	alpha = 1;
	cost0 = f(:)' * Q * f(:) + norm(tau)^2 * param.r; %u' * R * u
	while 1
		tautmp = tau + dtau * alpha;
		utmp = kron(eye(param.nbData-1),inv(L')) * tautmp; %Generalized forces
		xabstmp = fwdDyn(x_abs0, utmp, param); %Absolute angles
		xtmp = inv(blkdiag(L,L)) * xabstmp; %Relative angles
		ftmp = f_reach(xtmp(1:param.nbVarX,tl), param);
		cost = ftmp(:)' * Q * ftmp(:) + norm(tautmp)^2 * param.r; %utmp' * R * utmp
		if cost < cost0 || alpha < 1E-3
			break;
		end
		alpha = alpha * 0.5;
	end
	tau = tau + dtau * alpha;

	if norm(dtau * alpha) < 1E-2
		break; %Stop iLQR when solution is reached
	end
end
disp(['iLQR converged in ' num2str(n) ' iterations.']);

%Log data
r.x = x;
r.f = fkin(x(1:param.nbVarX,:), param); 
r.tau = reshape(u, param.nbVarU, param.nbData-1);

%% Plot state space
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
msh0 = diag(param.sz) * [-1 -1 1 1 -1; -1 1 1 -1 -1];
for t=1:param.nbPoints
	msh(:,:,t) = param.A(:,:,t) * msh0 + repmat(param.Mu(1:2,t), 1, size(msh0,2));
end

h = figure('position',[10,10,800,800],'color',[1,1,1]); hold on; axis off;
ftmp = fkin0(x(1:param.nbVarX,1), param);
plot(ftmp(1,:), ftmp(2,:), '-','linewidth',4,'color',[.8 .8 .8]);
ftmp = fkin0(x(1:param.nbVarX,tl(end)), param);
plot(ftmp(1,:), ftmp(2,:), '-','linewidth',4,'color',[.4 .4 .4]);
colMat = lines(param.nbPoints);
for t=1:param.nbPoints
	patch(msh(1,:,t), msh(2,:,t), min(colMat(t,:)*1.7,1),'linewidth',2,'edgecolor',colMat(t,:));
end
plot(r.f(1,:), r.f(2,:), '-','linewidth',2,'color',[0 0 0]);
plot(r.f(1,1), r.f(2,1), '.','markersize',40,'color',[0 0 0]);
plot(r.f(1,tl), r.f(2,tl), '.','markersize',30,'color',[0 0 0]);
axis equal; 

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
	T = tril(ones(size(x,1)));
	f = [param.l * cos(T * x); ...
		 param.l * sin(T * x); ...
		 mod(sum(x,1)+pi, 2*pi) - pi]; %x1,x2,o (orientation as single Euler angle for planar robot)
end

%%%%%%%%%%%%%%%%%%%%%%
%Forward kinematics for all robot articulations (in robot coordinate system)
function f = fkin0(x, param)
	L = tril(ones(size(x,1)));
	L2 = tril(repmat(param.l, size(x,1), 1));
	f = [L2 * cos(L * x), ...
		 L2 * sin(L * x)]'; 
	f = [zeros(2,1), f];
end

%%%%%%%%%%%%%%%%%%%%%%
%Jacobian with analytical computation (for single time step)
function J = jkin(x, param)
	L = tril(ones(size(x,1)));
	J = [-sin(L * x)' * diag(param.l) * L; ...
		  cos(L * x)' * diag(param.l) * L; ...
		  ones(1, size(x,1))]; %x1,x2,o
end

%%%%%%%%%%%%%%%%%%%%%%
%Cost and gradient for a viapoints reaching task (in object coordinate system)
function [f, J] = f_reach(x, param)
% 	f = fkin(x, param) - param.Mu; %Error by ignoring manifold
	f = logmap(fkin(x, param), param.Mu); %Error by considering manifold
	
	J = []; 
	for t=1:size(x,2)
% 		f(:,t) = logmap(fkin(x(:,t), param), param.Mu(:,t));
		f(1:2,t) = param.A(:,:,t)' * f(1:2,t); %Object-centered FK
		
		Jtmp = jkin(x(:,t), param);
		Jtmp(1:2,:) = param.A(:,:,t)' * Jtmp(1:2,:); %Object-centered Jacobian
		
		%Bounding boxes (optional)
		for i=1:2
			if abs(f(i,t)) < param.sz(i)
				f(i,t) = 0;
				Jtmp(i,:) = 0;
			else
				f(i,t) = f(i,t) - sign(f(i,t)) * param.sz(i);
			end
		end
		
		J = blkdiag(J, Jtmp);
	end
end

%%%%%%%%%%%%%%%%%%%%%%
%Forward dynamic to compute 
function [x, Su] = fwdDyn(x, u,param)
	g = 9.81; %Gravity norm
	kv = 1; %Joints damping
	l = param.l;
	nbDOFs = length(l);
	nbData = (size(u,1) / nbDOFs) + 1;
	Lm = triu(ones(nbDOFs)) .* repmat(param.m, nbDOFs, 1);
	L = tril(ones(nbDOFs));
	Su = zeros(2*nbDOFs*nbData, nbDOFs*(nbData-1));

	%Precomputation of mask (in tensor form)
	for j=1:nbDOFs
		J_index = zeros(1, nbDOFs);
		J_index(1,j) = 1;
		S1(:,:,j) = repmat(J_index * eye(nbDOFs), nbDOFs, 1) - repmat(eye(nbDOFs) * J_index', 1, nbDOFs);
	end
	
	
	for t=1:nbData-1	
		%Computation in matrix form of J, G, M, C
		G = -sum(Lm,2) .* l' .* cos(x(1:nbDOFs, t)) * g;
		C = -(l' * l) .* sin(x(1:nbDOFs,t) - x(1:nbDOFs,t)') .* (Lm.^.5 * Lm.^.5');
		M = (l' * l) .* cos(x(1:nbDOFs,t) - x(1:nbDOFs,t)') .* (Lm.^.5 * Lm.^.5');
		
		%Computation in tensor form of derivatives dJ, dG, dM, dC
		dG = diag(sum(Lm,2) .* l' .* sin(x(1:nbDOFs,t)) * g);
		dM_tmp = (l' * l) .* sin(x(1:nbDOFs,t) - x(1:nbDOFs,t)') .* (Lm.^.5 * Lm.^.5');
		dM = repmat(dM_tmp, [1, 1, nbDOFs]) .* S1;
		dC_tmp = (l' * l) .* cos(x(1:nbDOFs,t) - x(1:nbDOFs,t)') .* (Lm.^.5 * Lm.^.5');
		dC = repmat(dC_tmp, [1, 1, nbDOFs]) .* S1;

		%Update pose
		u_t = u((t-1) * nbDOFs + 1:t * nbDOFs);
		ddx = (M) \ (u_t + G + C * (x(nbDOFs+1:2*nbDOFs,t)).^2 - inv(L') * x(nbDOFs + 1:2 * nbDOFs,t) * kv); %With external force and joint damping
		x(:,t+1) = x(:,t) + [x(nbDOFs + 1:2 * nbDOFs, t); ddx] * param.dt;
		
		%Compute local linear systems
		invM = inv(M);
		for j=1:nbDOFs
			A21(:,j) = -invM * dM(:,:,j) * invM * u_t - invM * dM(:,:,j) * invM * G + ...
			            invM * dG(:,j) - invM * dM(:,:,j) * invM  * C * (x(nbDOFs + 1:2 * nbDOFs, t)).^2 + ...
			            invM * dC(:,:,j) * (x(nbDOFs + 1:2 * nbDOFs,t)).^2 + ...
						invM * dM(:,:,j) * invM * inv(L') * x(nbDOFs + 1:2 * nbDOFs,t) * kv;
		end
		
		%Linear systems with all components
		A(:,:,t) = [eye(nbDOFs), eye(nbDOFs) * param.dt; A21 * param.dt, eye(nbDOFs) + ...
		            (invM * (2 * C * diag([x(nbDOFs + 1:2 * nbDOFs, t)])- inv(L') * kv)) * param.dt];
		B(:,:,t) = [zeros(nbDOFs); invM * param.dt];
		
		Su(2 * nbDOFs * t + 1:2 * nbDOFs * (t + 1),:) = A(:,:,t) * Su(2 * nbDOFs * (t - 1) + 1:2 * nbDOFs * t,:);
		Su(2 * nbDOFs * t + 1:2 * nbDOFs * (t + 1), nbDOFs * (t - 1) + 1:nbDOFs * t) = B(:,:,t);
	end %t
end
