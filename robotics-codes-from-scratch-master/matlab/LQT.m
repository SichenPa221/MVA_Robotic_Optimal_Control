%% Linear quadratic tracking (LQT) applied to a viapoint task (batch formulation)
%% 
%% Copyright (c) 2023 Idiap Research Institute <https://www.idiap.ch/>
%% Written by Sylvain Calinon <https://calinon.ch>
%% 
%% This file is part of RCFS <https://robotics-codes-from-scratch.github.io/>
%% License: MIT

function LQT

%% Parameters
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
param.nbData = 100; %Number of datapoints
param.nbPoints = 3; %Number of viapoints
param.nbVarPos = 2; %Dimension of position data (here: x1,x2)
param.nbDeriv = 2; %Number of static and dynamic features (nbDeriv=2 for [x,dx] and u=ddx)
param.nbVar = param.nbVarPos * param.nbDeriv; %Dimension of state vector
param.dt = 1E-2; %Time step duration
param.r = 1E-12; %Control cost in LQR (for other cases)

%Task setting (viapoints passing)
Mu = [rand(param.nbVarPos, param.nbPoints) - 0.5; zeros(param.nbVar-param.nbVarPos, param.nbPoints)]; %Viapoints

%Q = speye(param.nbVar * param.nbPoints) * 1E0; %Precision matrix (for full state)
Q = kron(eye(param.nbPoints), diag([ones(1,param.nbVarPos) * 1E0, zeros(1,param.nbVar-param.nbVarPos)])); %Precision matrix (for position only)

R = speye((param.nbData-1)*param.nbVarPos) * param.r; %Standard control weight matrix (at trajectory level)
%e = ones(param.nbData-1,1) * param.r;
%R = kron(spdiags([-e 2*e -e], -1:1, param.nbData-1, param.nbData-1), speye(param.nbVarPos)); %Control weight matrix as transition matrix to encourage smoothness

%Time occurrence of viapoints
tl = linspace(1, param.nbData, param.nbPoints+1);
tl = round(tl(2:end));
idx = (tl - 1) * param.nbVar + [1:param.nbVar]';

%Dynamical System settings (discrete version)
[Su0, Sx0] = computeSuSx(param);
Su = Su0(idx,:);
Sx = Sx0(idx,:);


%% LQT (batch formulation)
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
x0 = zeros(param.nbVar,1); %Initial state
u = (Su' * Q * Su + R) \ Su' * Q * (Mu(:) - Sx * x0); %Control commands
rx = reshape(Sx0*x0+Su0*u, param.nbVar, param.nbData); %Reproduced trajectory


%%% LQT with a new time step recomputed based on the maximum allowed velocity
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%dx_max = 5.0; %Allowed velocity limit
%[dx, id] = max(max(abs(rx(param.nbVarPos+1:2*param.nbVarPos,:)))); %Determine highest velocity
%s = dx / dx_max; %Compute scaling factor
%param.dt = param.dt * s; %New time step
%u = u / s^param.nbDeriv; %New control commands
%[Su0, Sx0] = computeSuSx(param); %New linear system
%rx = reshape(Sx0*x0+Su0*u, param.nbVar, param.nbData); %New reproduced trajectory


%% Plot 2D
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
h(1) = figure('position',[10 10 800 800]); hold on; axis off;
plot(rx(1,:), rx(2,:), 'k-','linewidth',2);
plot(rx(1,1), rx(2,1), 'k.','markersize',30);
plot(Mu(1,:), Mu(2,:), 'r.','markersize',30);
axis equal; 


%% Timeline plot
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
h(2) = figure('position',[950 10 800 800],'color',[1 1 1]); 
for j=1:param.nbVarPos
	subplot(param.nbVarPos,1,j); hold on;
	plot(rx(j,:), 'k-','linewidth',2);
	plot(rx(j,1), 'k.','markersize',30);
	plot(tl, Mu(j,:), 'r.','markersize',30);
	xlabel('t','fontsize',26); ylabel(['x_' num2str(j)],'fontsize',26);
end

waitfor(h);
end


function [Su0, Sx0] = computeSuSx(param)
    %Dynamical System settings (discrete version)
    A1d = zeros(param.nbDeriv);
    for i=0:param.nbDeriv-1
	    A1d = A1d + diag(ones(param.nbDeriv-i,1),i) * param.dt^i * 1/factorial(i); %Discrete 1D
    end
    B1d = zeros(param.nbDeriv,1); 
    for i=1:param.nbDeriv
	    B1d(param.nbDeriv-i+1) = param.dt^i * 1/factorial(i); %Discrete 1D
    end
    A = kron(A1d, speye(param.nbVarPos)); %Discrete nD
    B = kron(B1d, speye(param.nbVarPos)); %Discrete nD

    %Build Sx and Su transfer matrices
    Sx0 = kron(ones(param.nbData,1), speye(param.nbVar));
    Su0 = sparse(param.nbVar*param.nbData, param.nbVarPos*(param.nbData-1));
    M = B;
    for n=2:param.nbData
	    id1 = (n-1)*param.nbVar+1:param.nbData*param.nbVar;
	    Sx0(id1,:) = Sx0(id1,:) * A;
	    id1 = (n-1)*param.nbVar+1:n*param.nbVar; 
	    id2 = 1:(n-1)*param.nbVarPos;
	    Su0(id1,id2) = M;
	    M = [A*M(:,1:param.nbVarPos), M]; 
    end
end
