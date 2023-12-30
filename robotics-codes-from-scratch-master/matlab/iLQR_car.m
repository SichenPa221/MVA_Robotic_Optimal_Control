%% iLQR applied to a car parking problem
%% 
%% Copyright (c) 2023 Idiap Research Institute <https://www.idiap.ch/>
%% Written by Sylvain Calinon <https://calinon.ch>
%% 
%% This file is part of RCFS <https://robotics-codes-from-scratch.github.io/>
%% License: MIT

function iLQR_car

%% Parameters
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
param.dt = 1E-1; %Time step size
param.nbData = 100; %Number of datapoints
param.nbIter = 10; %Number of iterations for iLQR
param.nbPoints = 1; %Number of viapoints
param.nbVarX = 4; %Dimension of state (x1,x2,theta,phi)
param.nbVarU = 2; %Control space dimension (v,dphi)
param.l = 0.5; %Length of car
param.q = 1E3; %Precision weight 
param.r = 1E-6; %Control weight
param.Mu = [4; 3; pi/2; 0]; %Viapoint
% param.Mu = [.2; 2; pi/2; 0]; %Viapoint

Q = speye(param.nbVarX * param.nbPoints) * param.q; %Precision matrix
R = speye((param.nbData-1) * param.nbVarU) * param.r; %Control weight matrix 

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
    %Estimate step size with line search method
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
h = figure('position',[10,10,800,800]); hold on; axis off; 
for t=round(linspace(1, param.nbData, 20))
    R = [cos(r.x(3,t)) -sin(r.x(3,t)); sin(r.x(3,t)) cos(r.x(3,t))];
    msh = R * [-.6 -.6 .6 .6 -.6; -.4 .4 .4 -.4 -.4] + repmat(r.x(1:2,t),1,5);
    plot(msh(1,:), msh(2,:), '-','linewidth',2,'color',[.6 .6 .6]);
end
plot(r.x(1,:),r.x(2,:), '-','linewidth',2,'color',[0 0 0]);
%Initial pose
R = [cos(r.x(3,1)) -sin(r.x(3,1)); sin(r.x(3,1)) cos(r.x(3,1))];
msh = R * [-.6 -.6 .6 .6 -.6; -.4 .4 .4 -.4 -.4] + repmat(r.x(1:2,1),1,5);
plot(msh(1,:), msh(2,:), '-','linewidth',2,'color',[0 0 0]);
h(1) = plot(r.x(1,1),r.x(2,1), '.','markersize',20,'color',[0 0 0]);
%Target pose
R = [cos(param.Mu(3)) -sin(param.Mu(3)); sin(param.Mu(3)) cos(param.Mu(3))];
msh = R * [-.6 -.6 .6 .6 -.6; -.4 .4 .4 -.4 -.4] + repmat(param.Mu(1:2),1,5);
plot(msh(1,:), msh(2,:), '-','linewidth',2,'color',[.8 0 0]);
h(2) = plot(param.Mu(1), param.Mu(2), '.','markersize',20,'color',[.8 0 0]);
legend(h,{'Initial pose','Target pose'},'location','northwest','fontsize',30);
axis equal; axis([-1 5 -1 4]);

waitfor(h);
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
        dx(1) = cos(x(3,t)) * u(1,t);
        dx(2) = sin(x(3,t)) * u(1,t);
        dx(3) = tan(x(4,t)) * u(1,t) / param.l;
        dx(4) = u(2,t);
        x(:,t+1) = x(:,t) + dx * param.dt;
    end
end

%%%%%%%%%%%%%%%%%%%%%%
%Linearize the system along the trajectory computing the matrices A and B
function [A, B] = linSys(x, u, param)	
    A = zeros(param.nbVarX, param.nbVarX, param.nbData-1);
    B = zeros(param.nbVarX, param.nbVarU, param.nbData-1);
    Ac = zeros(param.nbVarX);
    Bc = zeros(param.nbVarX, param.nbVarU);
    for t=1:param.nbData-1
        %Linearize the system
        Ac(1,3) = -u(1,t) * sin(x(3,t));
        Ac(2,3) = u(1,t) * cos(x(3,t));
        Ac(3,4) = u(1,t) * tan(x(4,t)^2 + 1) / param.l;
        Bc(1,1) = cos(x(3,t)); 
        Bc(2,1) = sin(x(3,t)); 
        Bc(3,1) = tan(x(4,t)) / param.l;
        Bc(4,2) = 1;
        %Discretize the linear system
        A(:,:,t) = eye(param.nbVarX) + Ac * param.dt;
        B(:,:,t) = Bc * param.dt;
    end
end 
