%%    iLQR applied to a 2D point-mass system reaching a target while avoiding 
%%    obstacles represented as Gaussian process implicit surfaces (GPIS)
%%
%%    Note to run the code on Octave: This code uses the pdist2() function from the statistics package 
%%
%%    Copyright (c) 2021 Idiap Research Institute, http://www.idiap.ch/
%%    Written by Sylvain Calinon <https://calinon.ch>
%%
%%    This file is part of RCFS.
%%
%%    RCFS is free software: you can redistribute it and/or modify
%%    it under the terms of the GNU General Public License version 3 as
%%    published by the Free Software Foundation.
%%
%%    RCFS is distributed in the hope that it will be useful,
%%    but WITHOUT ANY WARRANTY; without even the implied warranty of
%%    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
%%    GNU General Public License for more details.
%%
%%    You should have received a copy of the GNU General Public License
%%    along with RCFS. If not, see <http://www.gnu.org/licenses/>.

function iLQR_obstacle_GPIS

if exist('OCTAVE_VERSION', 'builtin') ~= 0
	pkg('load', 'statistics');
end

%% Parameters
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
param.dt = 1E-2; %Time step size
param.nbData = 101; %Number of datapoints
param.nbIter = 300; %Maximum number of iterations for iLQR
param.nbPoints = 1; %Number of viapoints
param.nbVarX = 2; %State space dimension (x1,x2)
param.nbVarU = 2; %Control space dimension (dx1,dx2)
param.sz = [.2, .2]; %Size of objects
param.sz2 = [.4, .6]; %Size of obstacles
param.q = 1E2; %Tracking weight term
param.q2 = 1E0; %Obstacle avoidance weight term
param.r = 1E-3; %Control weight term

param.Mu = [.9; .9; pi/6]; %Viapoints (x1,x2,o)
for t=1:param.nbPoints
	param.A(:,:,t) = [cos(param.Mu(3,t)), -sin(param.Mu(3,t)); sin(param.Mu(3,t)), cos(param.Mu(3,t))]; %Orientation
end

Q = speye(param.nbVarX * param.nbPoints) * param.q; %Precision matrix to reach viapoints
R = speye((param.nbData-1) * param.nbVarU) * param.r; %Control weight matrix (at trajectory level)

%Time occurrence of viapoints
tl = linspace(1, param.nbData, param.nbPoints+1);
tl = round(tl(2:end));
idx = (tl - 1) * param.nbVarX + [1:param.nbVarX]';


%% GPIS representation of obstacles
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
param.p = [1.4, 1E-5]; %Thin-plate covariance function parameters 
param.x = [0.2, 0.4, 0.6, -0.4, 0.6, 0.9; ...
           0.5, 0.5, 0.5,  0.8, 0.1, 0.6];
param.y = [ -1,   0,   1,   -1,  -1,  -1];

%Disc as geometric prior
rc = 4E-1; %Radius of disc
xc = [.05; .05] + .5; %Location of disc
S = eye(2) * rc^-2;
param.Mu2 = .5 * rc * diag(1 - (param.x - repmat(xc,1,size(param.x,2)))' * S * (param.x - repmat(xc,1,size(param.x,2))))';
param.K = covFct(param.x, param.x, param.p, 1); %Inclusion of noise on the inputs for the computation of K


%% Iterative LQR (iLQR)
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
u = zeros(param.nbVarU*(param.nbData-1), 1); %Initial commands
x0 = [.3; .05]; %Initial state

%Transfer matrices (for linear system as single integrator)
Su0 = [zeros(param.nbVarX, param.nbVarX*(param.nbData-1)); tril(kron(ones(param.nbData-1), eye(param.nbVarX)*param.dt))];
Sx0 = kron(ones(param.nbData,1), eye(param.nbVarX));
Su = Su0(idx,:);

for n=1:param.nbIter
	x = reshape(Su0 * u + Sx0 * x0, param.nbVarX, param.nbData); %System evolution
	[f, J] = f_reach(x(:,tl), param); %Residuals and Jacobians (tracking objective)
	[f2, J2, id2] = f_avoid(x, param); %Residuals and Jacobians (avoidance objective)
	Su2 = Su0(id2,:);
	
	du = (Su' * J' * Q * J * Su + Su2' * J2' * J2 * Su2 * param.q2 + R) \ ...
		 (-Su' * J' * Q * f(:) - Su2' * J2' * f2(:) * param.q2 - u * param.r); %Gauss-Newton update
	
	%Estimate step size with backtracking line search method
	alpha = 1;
	cost0 = norm(f(:))^2 * param.q + norm(f2(:))^2 * param.q2 + norm(u)^2 * param.r; %Cost
	while 1
		utmp = u + du * alpha;
		xtmp = reshape(Su0 * utmp + Sx0 * x0, param.nbVarX, param.nbData);
		ftmp = f_reach(xtmp(:,tl), param); %Residuals (tracking objective)
		ftmp2 = f_avoid(xtmp, param); %Residuals (avoidance objective)
		cost = norm(ftmp(:))^2 * param.q + norm(ftmp2(:))^2 * param.q2 + norm(utmp)^2 * param.r; %Cost
		if cost < cost0 || alpha < 1E-3
			break;
		end
		alpha = alpha * 0.5;
	end
	u = u + du * alpha;
	
	if norm(du * alpha) < 5E-2
		break; %Stop iLQR when solution is reached
	end
end
disp(['iLQR converged in ' num2str(n) ' iterations.']);


%% Plot state space
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
limAxes = [0 1 0 1];
colMat = lines(param.nbPoints); %Colors for viapoints

nbRes = 30;
[xx, yy] = ndgrid(linspace(limAxes(1),limAxes(2),nbRes), linspace(limAxes(3),limAxes(4),nbRes));
xtmp = [xx(:)'; yy(:)'];
msh = [min(xtmp(1,:)), min(xtmp(1,:)), max(xtmp(1,:)), max(xtmp(1,:)), min(xtmp(1,:)); ...
       min(xtmp(2,:)), max(xtmp(2,:)), max(xtmp(2,:)), min(xtmp(2,:)), min(xtmp(2,:))];
%Avoidance
[f2,~,~,idt] = f_avoid(xtmp, param);
z2 = zeros(nbRes^2,1);
z2(idt) = f2.^2;
zz2 = reshape(z2, nbRes, nbRes);

h = figure('position',[10,10,1400,1400],'color',[1,1,1]); hold on; axis off; rotate3d on;
colormap(repmat(linspace(1,.4,64),3,1)');
plot(msh(1,:), msh(2,:),'-','linewidth',1,'color',[0 0 0]); %border
plot(param.Mu(1,:), param.Mu(2,:), '.','markersize',40,'color',[.8 0 0]); %Viapoints
plot3(x(1,[1,end]), x(2,[1,end]), zeros(1,2), '-','linewidth',2,'color',[.7 .7 .7]); %Initialization
plot(x(1,:), x(2,:), '-','linewidth',2,'color',[0 0 0]);
plot(x(1,[1,tl(1:end-1)]), x(2,[1,tl(1:end-1)]), '.','markersize',30,'color',[0 0 0]);
axis equal; axis(limAxes); %view(0,90); %axis vis3d;
surface(xx, yy, zz2-max(zz2(:)), 'EdgeColor','interp','FaceColor','interp');

waitfor(h);
end 


%%%%%%%%%%%%%%%%%%%%%%
%Residuals f and Jacobians J for a viapoints reaching task (in object coordinate system)
function [f, J] = f_reach(x, param)
	f = x - param.Mu(1:2,:); %Error by ignoring manifold
	J = eye(param.nbVarX * size(x,2)); %Jacobian
end

%%%%%%%%%%%%%%%%%%%%%%%%%
%Error function
function d = substr(x1, x2)
	d = x1 - x2;
end

%%%%%%%%%%%%%%%%%%%%%%%%%
%Covariance function in GPIS
function [K, dK] = covFct(x1, x2, p, flag_noiseObs)
	if nargin<4
		flag_noiseObs = 0;
	end
	
	%Thin plate covariance function (for 3D implicit shape)
	K = 12^-1 * (2 * pdist2(x1',x2').^3 - 3 * p(1) * pdist2(x1',x2').^2 + p(1)^3); %Kernel
	dK(:,:,1) = 12^-1 * (6 * pdist2(x1',x2') .* substr(x1(1,:)',x2(1,:)) - 6 * p(1) * substr(x1(1,:)',x2(1,:))); %Derivatives along x1
	dK(:,:,2) = 12^-1 * (6 * pdist2(x1',x2') .* substr(x1(2,:)',x2(2,:)) - 6 * p(1) * substr(x1(2,:)',x2(2,:))); %Derivatives along x2

% 	%RBF covariance function
% 	p = [5E-2^-1, 1E-4, 1E-2];
% 	K = p(3) * exp(-p(1) * pdist2(x1', x2').^2); %Kernel
% 	dK(:,:,1) = -p(1) * p(3) * exp(-p(1) * pdist2(x1', x2').^2) .* substr(x1(1,:)',x2(1,:)); %Derivatives along x1
% 	dK(:,:,2) = -p(1) * p(3) * exp(-p(1) * pdist2(x1', x2').^2) .* substr(x1(2,:)',x2(2,:)); %Derivatives along x2
	
	if flag_noiseObs==1
		K = K + p(2) * eye(size(x1,2),size(x2,2)); %Consideration of noisy observation y
	end
end

%%%%%%%%%%%%%%%%%%%%%%%%%
%Residuals f and Jacobians J for obstacle avoidance with GPIS representation (for a given time step)
function [f, J] = GPIS(x, param)
	[K, dK] = covFct(x, param.x, param.p);
	f = (K / param.K * param.y')'; %GPR with Mu=0
	% f = param.MuS + (K / param.K * (param.y - param.Mu2)')';
	J = [(dK(:,:,1) / param.K * (param.y - param.Mu2)')'; (dK(:,:,2) / param.K * (param.y - param.Mu2)')']; %Gradients

	%Reshape gradients
	a = max(f, 0); %Amplitude
	J = 1E2 * repmat(tanh(a), [2,1]) .* J ./ repmat(sum(J.^2).^.5, [2,1]); %Vector moving away from interior of shape
end

%%%%%%%%%%%%%%%%%%%%%%
%Residuals f and Jacobians J for obstacle avoidance with GPIS representation (for all time steps)
function [f, J, id, idt] = f_avoid(x, param)
	f=[]; J=[]; id=[]; idt=[];
	[ftmp, Jtmp] = GPIS(x, param);
	for t=1:size(x,2)
		%Bounding boxes
		if ftmp(t) > 0
			f = [f; ftmp(t)];
			J = blkdiag(J, Jtmp(:,t)');
			id = [id, (t-1) * param.nbVarU + [1:param.nbVarU]];
			idt = [idt, t];
		end
	end
end
