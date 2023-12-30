%% Point-mass LQR with infinite horizon
%% 
%% Copyright (c) 2023 Idiap Research Institute <https://www.idiap.ch/>
%% Written by Sylvain Calinon <https://calinon.ch>
%% 
%% This file is part of RCFS <https://robotics-codes-from-scratch.github.io/>
%% License: MIT


function LQR_infHor
    
%% Parameters
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
param.nbData = 200; %Number of datapoints
param.nbRepros = 4; %Number of reproductions

param.nbVarPos = 2; %Dimension of position data (here: x1,x2)
param.nbDeriv = 2; %Number of static & dynamic features (D=2 for [x,dx])
param.nbVar = param.nbVarPos * param.nbDeriv; %Dimension of state vector in the tangent space
param.dt = 1E-2; %Time step duration
param.rfactor = 4E-2;	%Control cost in LQR 

%Control cost matrix
R = eye(param.nbVarPos) * param.rfactor;

%Target and desired covariance
param.Mu = [randn(param.nbVarPos,1); zeros(param.nbVarPos*(param.nbDeriv-1),1)];

[Ar,~] = qr(randn(param.nbVarPos));
xCov = Ar * diag(rand(param.nbVarPos,1)) * Ar' * 1E-1

%% Discrete dynamical System settings 
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
A1d = zeros(param.nbDeriv);
for i=0:param.nbDeriv-1
    A1d = A1d + diag(ones(param.nbDeriv-i,1),i) * param.dt^i * 1/factorial(i); %Discrete 1D
end
B1d = zeros(param.nbDeriv,1); 
for i=1:param.nbDeriv
    B1d(param.nbDeriv-i+1) = param.dt^i * 1/factorial(i); %Discrete 1D
end
A = kron(A1d, eye(param.nbVarPos)); %Discrete nD
B = kron(B1d, eye(param.nbVarPos)); %Discrete nD


%% discrete LQR with infinite horizon
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
Q = blkdiag(inv(xCov), zeros(param.nbVarPos*(param.nbDeriv-1))); %Precision matrix
P = solveAlgebraicRiccati_eig_discrete(A, B*(R\B'), (Q+Q')/2);
L = (B' * P * B + R) \ B' * P * A; %Feedback gain (discrete version)

for n=1:param.nbRepros
    x = [ones(param.nbVarPos,1)+randn(param.nbVarPos,1)*5E-1; zeros(param.nbVarPos*(param.nbDeriv-1),1)];
    for t=1:param.nbData		
        r(n).Data(:,t) = x; 
        u = L * (param.Mu - x); %Compute acceleration (with only feedback terms)
        x = A * x + B * u;
    end
end


%% Plots
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%


figure('position',[10,10,650,650]); hold on; axis off; grid off; 
plotGMM(param.Mu(1:2), xCov(1:2,1:2), [.8 0 0], .3);
for n=1:param.nbRepros
    plot(r(n).Data(1,:), r(n).Data(2,:), '-','linewidth',1,'color',[0 0 0]);
end
%plot(param.Mu(1,1), param.Mu(2,1), 'r.','markersize',80);
axis equal; 
%print('-dpng','graphs/demo_MPC_infHor01.png');


%% Timeline plot
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

labList = {'x_1','x_2','dx_1','dx_2','ddx_1','ddx_2'};
figure('position',[720 10 600 650],'color',[1 1 1]); 
for j=1:param.nbVar
subplot(param.nbVar+1,1,j); hold on;
for n=1:param.nbRepros
    plot(r(n).Data(j,:), '-','linewidth',1,'color',[0 0 0]);
end
if j<7
    ylabel(labList{j},'fontsize',14,'interpreter','latex');
end
end
%Speed profile
if param.nbDeriv>1
subplot(param.nbVar+1,1,param.nbVar+1); hold on;
for n=1:param.nbRepros
    sp = sqrt(r(n).Data(3,:).^2 + r(n).Data(4,:).^2);
    plot(sp, '-','linewidth',1,'color',[0 0 0]);
end
ylabel('|dx|','fontsize',14);
xlabel('t','fontsize',14);
end
%print('-dpng','graphs/demo_LQR_infHor01.png');

pause;
close all;
end

%%%%%%%%%%%%%%%%%%%%%%
% Solve Algebraic Ricatty discrete equation
function X = solveAlgebraicRiccati_eig_discrete(A, G, Q)
    n = size(A,1);

    %Symplectic matrix (see https://en.wikipedia.org/wiki/Algebraic_Riccati_equation)
    %Z = [A+B*(R\B')/A'*Q, -B*(R\B')/A'; -A'\Q, A'^-1]; 
    Z = [A+G/A'*Q, -G/A'; -A'\Q, inv(A')];


    %Since Z is symplectic, if it does not have any eigenvalues on the unit circle, 
    %then exactly half of its eigenvalues are inside the unit circle. 
    [V,D] = eig(Z);
    U = [];
    for j=1:2*n
        if norm(D(j,j)) < 1 %inside unit circle
            U = [U V(:,j)];
        end
    end

    X = real(U(n+1:end,:) / U(1:n,:));
end

%%%%%%%%%%%%%%%%%%%%%%
% Plot a 2D Mixture of Gaussians
function [h, X] = plotGMM(Mu, Sigma, color, valAlpha)
    nbStates = size(Mu,2);
    nbDrawingSeg = 100;
    darkcolor = color * .7; %max(color-0.5,0);
    t = linspace(-pi, pi, nbDrawingSeg);
    if nargin<4
        valAlpha = 1;
    end

    h = [];
    X = zeros(2,nbDrawingSeg,nbStates);
    for i=1:nbStates
        [V,D] = eig(Sigma(:,:,i));
        R = real(V*D.^.5);
        X(:,:,i) = R * [cos(t); sin(t)] + repmat(Mu(:,i), 1, nbDrawingSeg);
        if nargin>3 %Plot with alpha transparency
            h = [h patch(X(1,:,i), X(2,:,i), color, 'lineWidth', 1, 'EdgeColor', color, 'facealpha', valAlpha,'edgealpha', valAlpha)];
            h = [h plot(Mu(1,:), Mu(2,:), '.', 'markersize', 10, 'color', darkcolor)];
        else %Plot without transparency
            h = [h patch(X(1,:,i), X(2,:,i), color, 'lineWidth', 1, 'EdgeColor', darkcolor)];
            h = [h plot(Mu(1,:), Mu(2,:), '.', 'markersize', 10, 'color', darkcolor)];
        end
    end
end



