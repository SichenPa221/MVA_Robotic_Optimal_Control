%% iLQR applied to a bicopter problem
%% 
%% Copyright (c) 2023 Idiap Research Institute <https://www.idiap.ch/>
%% Written by Sylvain Calinon <https://calinon.ch>
%% 
%% This file is part of RCFS <https://robotics-codes-from-scratch.github.io/>
%% License: MIT

function iLQR_bicopter

%% Parameters
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

param.dt = 5E-2; %Time step size
param.nbData = 100; %Number of datapoints
param.nbIter = 20; %Maximum number of iterations for iLQR
param.nbPoints = 1; %Number of viapoints
param.nbVarPos = 3; %Dimension of position (x1,x2,theta)
param.nbDeriv = 2; %Number of derivatives (nbDeriv=2 for [x; dx] state)
param.nbVarX = param.nbVarPos * param.nbDeriv; %State space dimension
param.nbVarU = 2; %Control space dimension (u1,u2)

param.l = 0.5; %Length of the bicopter
param.m = 1.5; %Mass of the bicopter
param.I = 2; %param.m * param.l^2 / 12; %Inertia (I=m*l^2/12)
param.g = 9.81; %Acceleration due to gravity
param.q = 1E3; %Precision weight 
param.r = 1E-3; %Control weight
param.Mu = [4; 4; 0; zeros(param.nbVarPos,1)]; %Target

Q = speye(param.nbVarX * param.nbPoints) * param.q; %Precision matrix
R = speye((param.nbData-1)*param.nbVarU) * param.r; %Control weight matrix 

%Time occurrence of viapoints
tl = linspace(1, param.nbData, param.nbPoints+1);
tl = round(tl(2:end));
idx = (tl - 1) * param.nbVarX + [1:param.nbVarX]';


%% Iterative LQR (iLQR)
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
u = zeros(param.nbVarU*(param.nbData-1), 1);
x0 = zeros(param.nbVarX, 1);

for n=1:param.nbIter
    %System evolution
    x = dynSysSimulation(x0, reshape(u, param.nbVarU, param.nbData-1), param);
    %Linearization
    [A, B] = linSys(x, reshape(u, param.nbVarU, param.nbData-1), param);
    Su0 = transferMatrices(A, B);
    Su = Su0(idx,:);
    %Gauss-Newton update
    e = param.Mu - x(:,tl);
    du = (Su' * Q * Su + R) \ (Su' * Q * e(:) - R * u);
    %Estimate step size with backtracking line search method
    alpha = 1;
    cost0 = e(:)' * Q * e(:) + u' * R * u;
    while 1
        utmp = u + du * alpha;
        xtmp = dynSysSimulation(x0, reshape(utmp, param.nbVarU, param.nbData-1), param);
        etmp = param.Mu - xtmp(:,tl);
        cost = etmp(:)' * Q * etmp(:) + utmp' * R * utmp; %Compute the cost
        if cost < cost0 || alpha < 1E-4
            break;
        end
        alpha = alpha * 0.5;
    end
    u = u + du * alpha; %Update control by following gradient
end

%Log data
r.x = x;
r.u = reshape(u, param.nbVarU, param.nbData-1);


%% Plot state space
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
h = figure('position',[10,10,800,800]); hold on; axis off; grid on; box on;
for t=floor(linspace(1,param.nbData,20))
    plot(r.x(1,t), r.x(2,t), '.','markersize',40,'color',[.6 .6 .6]);
    msh = [r.x(1:2,t), r.x(1:2,t)] + [cos(r.x(3,t)); sin(r.x(3,t))] * [-.4, .4];
    plot(msh(1,:), msh(2,:), '-','linewidth',5,'color',[.6 .6 .6]);
end
plot(r.x(1,:),r.x(2,:), '-','linewidth',2,'color',[.6 .6 .6]);
msh = [r.x(1:2,1), r.x(1:2,1)] + [cos(r.x(3,1)); sin(r.x(3,1))] * [-.4, .4];
plot(msh(1,:), msh(2,:), '-','linewidth',5,'color',[0 0 0]);
h(1) = plot(r.x(1,1),r.x(2,1), '.','markersize',40,'color',[0 0 0]);
msh = [param.Mu(1:2), param.Mu(1:2)] + [cos(param.Mu(3)); sin(param.Mu(3))] * [-.4, .4];
plot(msh(1,:), msh(2,:), '-','linewidth',5,'color',[.8 0 0]);
h(2) = plot(param.Mu(1), param.Mu(2), '.','markersize',40,'color',[.8 0 0]);
legend(h,{'Initial pose','Target pose'},'location','northwest','fontsize',30);
axis equal; %axis([-.5 4.5 -.2 4.4]);
xlabel('x_1'); ylabel('x_2');

waitfor(h);
close all;
end 
    
%%%%%%%%%%%%%%%%%%%%%%
function [Su, Sx] = transferMatrices(A, B)
    [nbVarX, nbVarU, nbData] = size(B);
    nbData = nbData+1;
    Sx = kron(ones(nbData,1), speye(nbVarX)); 
    Su = sparse(zeros(nbVarX*(nbData-1), nbVarU*(nbData-1)));
    for t=1:nbData-1
        id1 = (t-1)*nbVarX+1:t*nbVarX;
        id2 = t*nbVarX+1:(t+1)*nbVarX;
        id3 = (t-1)*nbVarU+1:t*nbVarU;
        Sx(id2,:) = squeeze(A(:,:,t)) * Sx(id1,:);
        Su(id2,:) = squeeze(A(:,:,t)) * Su(id1,:);	
        Su(id2,id3) = B(:,:,t);	
    end
end

%%%%%%%%%%%%%%%%%%%%%%
%Given the control trajectory u and initial state x0, compute the whole state trajectory
function x = dynSysSimulation(x0, u, param)	
    x = zeros(param.nbVarX, param.nbData);
    dx = zeros(param.nbVarX, 1);
    x(:,1) = x0;
    for t=1:param.nbData-1
        dx(1) = x(4,t);
        dx(2) = x(5,t);
        dx(3) = x(6,t);
        dx(4) = -(u(1,t) + u(2,t)) * sin(x(3,t)) / param.m;
        dx(5) =  (u(1,t) + u(2,t)) * cos(x(3,t)) / param.m - param.g;
        dx(6) =  (u(1,t) - u(2,t)) * param.l / param.I;
        x(:,t+1) = x(:,t) + dx * param.dt;
    end
end

%%%%%%%%%%%%%%%%%%%%%%
%Linearize the system along the trajectory computing the matrices A and B
function [A, B] = linSys(x, u, param)
    A = zeros(param.nbVarX, param.nbVarX, param.nbData-1);
    B = zeros(param.nbVarX, param.nbVarU, param.nbData-1);
    Ac = zeros(param.nbVarX);
    Ac(1:3,4:6) = eye(param.nbVarPos);
    Bc = zeros(param.nbVarX, param.nbVarU);
    for t=1:param.nbData-1
        %Linearize the system
        Ac(4,3) = -(u(1,t) + u(2,t)) * cos(x(3,t)) / param.m;
        Ac(5,3) = -(u(1,t) + u(2,t)) * sin(x(3,t)) / param.m;
        Bc(4,1) = -sin(x(3,t)) / param.m;  Bc(4,2) =  Bc(4,1);
        Bc(5,1) =  cos(x(3,t)) / param.m;  Bc(5,2) =  Bc(5,1);
        Bc(6,1) =  param.l / param.I;      Bc(6,2) = -Bc(6,1);
        %Discretize the linear system
        A(:,:,t) = eye(param.nbVarX) + Ac * param.dt;
        B(:,:,t) = Bc * param.dt;
    end
end
    
