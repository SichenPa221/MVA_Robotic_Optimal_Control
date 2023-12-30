%% Batch LQT with nullspace formulation
%% 
%% Copyright (c) 2023 Idiap Research Institute <https://www.idiap.ch/>
%% Written by Sylvain Calinon <https://calinon.ch>
%% 
%% This file is part of RCFS <https://robotics-codes-from-scratch.github.io/>
%% License: MIT

function LQT_nullspace

%% Parameters
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
param.nbData = 50; %Number of datapoints
param.nbRepros = 60; %Number of stochastic reproductions
param.nbPoints = 1; %Number of keypoints
param.nbVar = 2; %Dimension of state vector
param.dt = 1E-1; %Time step duration
param.rfactor = 1E-4; %param.dt^nbDeriv;	%Control cost in LQR


R = speye((param.nbData-1)*param.nbVar) * param.rfactor;
x0 = zeros(param.nbVar,1); %Initial point


%% Dynamical System settings
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
Su = [zeros(param.nbVar, param.nbVar*(param.nbData-1)); kron(tril(ones(param.nbData-1)), eye(param.nbVar)*param.dt)];
Sx = kron(ones(param.nbData,1), eye(param.nbVar));


%% Task setting
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
tl = linspace(1,param.nbData,param.nbPoints+1);
tl = round(tl(2:end)); %[param.nbData/2, param.nbData];
%Mu = rand(param.nbVarPos,param.nbPoints) - 0.5; 
Mu = [20; 10];

Sigma = repmat(eye(param.nbVar)*1E-3, [1,1,param.nbPoints]);
MuQ = zeros(param.nbVar*param.nbData,1); 
Q = zeros(param.nbVar*param.nbData);
for t=1:length(tl)
    id(:,t) = [1:param.nbVar] + (tl(t)-1) * param.nbVar;
    Q(id(:,t), id(:,t)) = inv(Sigma(:,:,t));
    MuQ(id(:,t)) = Mu(:,t);
end


%% Batch LQR reproduction
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%Precomputation of basis functions to generate control commands u (here, with Bernstein basis functions)
nbRBF = 10;
H = buildPhiBernstein(param.nbData-1,nbRBF);

%Reproduction with nullspace planning
[V,D] = eig(Q);
U = V * D.^.5;
J = U' * Su; %Jacobian

%Left pseudoinverse solution
pinvJ = (J' * J + R) \ J'; %Left pseudoinverse
N = speye((param.nbData-1)*param.nbVar) - pinvJ * J; %Nullspace projection matrix
u1 = pinvJ * U' * (MuQ - Sx * x0); %Principal task (least squares solution)
x = reshape(Sx*x0+Su*u1, param.nbVar, param.nbData); %Reshape data for plotting

%General solutions
for n=1:param.nbRepros
    w = randn(param.nbVar,nbRBF) * 1E1; %Random weights	
    u2 = w * H'; %Reconstruction of control signals by a weighted superposition of basis functions
    u = u1 + N * u2(:);
    r(n).x = reshape(Sx*x0+Su*u, param.nbVar, param.nbData); %Reshape data for plotting
end


%% Plot 2D
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
figure('position',[10 10 800 800],'color',[1 1 1],'name','x1-x2 plot'); hold on; axis off;
for n=1:param.nbRepros
    plot(r(n).x(1,:), r(n).x(2,:), '-','linewidth',1,'color',.9-[.7 .7 .7].*rand(1));
end
plot(x(1,:), x(2,:), '-','linewidth',2,'color',[.8 0 0]);
plot(x(1,1), x(2,1), 'o','linewidth',2,'markersize',8,'color',[.8 0 0]);
plot(MuQ(id(1,:)), MuQ(id(2,:)), '.','markersize',30,'color',[.8 0 0]);
axis equal; 
%print('-dpng','graphs/LQT_nullspace02.png');

pause(10);
close all;
end

%Building Bernstein basis functions
function phi = buildPhiBernstein(nbData, nbFct)
	t = linspace(0, 1, nbData);
	phi = zeros(nbData, nbFct);
	for i=1:nbFct
		phi(:,i) = factorial(nbFct-1) ./ (factorial(i-1) .* factorial(nbFct-i)) .* (1-t).^(nbFct-i) .* t.^(i-1);
	end
end

