function spline2D
%% Concatenated Bernstein basis functions with constraints to encode
%% a signed distance function (2D inputs, 1D output)
%% 
%% Copyright (c) 2023 Idiap Research Institute <https://www.idiap.ch/>
%% Written by Sylvain Calinon <https://calinon.ch>
%% 
%% This file is part of RCFS <https://robotics-codes-from-scratch.github.io/>
%% License: MIT


%% Parameters
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
nbFct = 5; %Number of basis functions for each dimension 
nbSeg = 4; %Number of segments for each dimension 
nbIn = 2; %Dimension of input data (here: 2D surface embedded in 3D)
nbOut = 1; %Dimension of output data (here: height data)
nbDim = 40; %Grid size for each dimension

%Reference surface x0
load('../data/sdf01.mat');

%Input array
[T1, T2] = meshgrid(linspace(0,1,nbDim), linspace(0,1,nbDim));
t12 = [T1(:)'; T2(:)'];


%% Bézier curve in matrix form
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%Time parameters matrix to compute positions
nbT = nbDim / nbSeg;
t = linspace(0, 1-1/nbT, nbT);
T0 = zeros(length(t), nbFct); 
for n=1:nbFct
	T0(:,n) = t.^(n-1);
end
%Bézier curve 
B0 = zeros(nbFct); %Polynomial coefficients matrix
for n=1:nbFct
	for i=1:nbFct
		B0(nbFct-i+1,n) = (-1)^(nbFct-i-n) * -binomial(nbFct-1, i-1) * binomial((nbFct-1)-(i-1), (nbFct-1)-(n-1)-(i-1));
	end
end
%Matrices for a concatenation of curves
T = [];
B = [];
for n=1:nbSeg
	T = blkdiag(T, T0);
	B = blkdiag(B, B0);
end

%Bézier curve constraint: Last control point and first control point of next segment should be the same (w4-w5=0, ...), 
%and the two control points around should be symmetric (w3-2*w5+w6=0, ...)
C0 = blkdiag(eye(nbFct-4), [[1; 0; 0; -1], [0; 1; 1; 2]]); %w4=w5 and w6=-w3+2*w5
C = eye(2);
for n=1:nbSeg-1
	C = blkdiag(C, C0);
end
C = blkdiag(C, eye(nbFct-2));

%1D primitive with constraint
phi = T * B * C; 

%Transform to multidimensional basis functions
Psi = kron(phi, phi); %Surface primitive


%% Encoding and reproduction with basis functions
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%Batch estimation of superposition weights from reference surface (here, with all points)
wb = Psi \ x0; 
%Reconstruction of surface from full observation
xb = Psi * wb; 

%% Plot
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
h = figure('position',[10 10 2400 1000]); 
colormap(repmat(linspace(1,.4,64),3,1)');

%Reference surface
subplot(1,2,1); hold on; axis off; %title('Reference SDF');
surface(T1, T2, reshape(x0, [nbDim, nbDim])-max(x0), 'FaceColor','interp','EdgeColor','interp');
contour(T1, T2, reshape(x0, [nbDim, nbDim]), [0:.02:1], 'linewidth',2);
msh = contour(T1, T2, reshape(x0, [nbDim, nbDim]), [0,0]);
plot(msh(1,2:end), msh(2,2:end), '-','linewidth',4,'color',[0 0 .8]);
axis tight; axis equal; axis([0 1 0 1]); axis ij; 

%Reconstructed surface
subplot(1,2,2); hold on; axis off; %title('Reconstructed SDF');
surface(T1, T2, reshape(xb, [nbDim, nbDim])-max(xb), 'FaceColor','interp','EdgeColor','interp'); %SDF
%surface(T1, T2, reshape(diag_Sigma_x, [nbDim, nbDim])-max(diag_Sigma_x), 'FaceColor','interp','EdgeColor','interp'); %Uncertainty
contour(T1, T2, reshape(xb, [nbDim, nbDim]), [0:.02:1], 'linewidth',2); %'color',[0 0 0]
msh = contour(T1, T2, reshape(xb, [nbDim, nbDim]), [0, 0]);
plot(msh(1,2:end), msh(2,2:end), '-','linewidth',4,'color',[0 0 .8]);

%Grid of control points
ttmp = linspace(0, 1, length(wb)^.5+nbSeg-1);
[T1tmp, T2tmp] = meshgrid(ttmp, ttmp);
plot(T1tmp, T2tmp, '.','markersize',12,'color',[.3,.3,.3]);

%Grid of patches
ttmp = linspace(0, 1, nbSeg+1);
[T1tmp, T2tmp] = meshgrid(ttmp, ttmp);
plot(T1tmp, T2tmp, '.','markersize',26,'color',[0,0,0]);
axis tight; axis equal; axis([0 1 0 1]); axis ij; 

waitfor(h);
end


%%%%%%%%%%%%%%%%
function b = binomial(n, i)
	if n>=0 && i>=0
		b = factorial(n) / (factorial(i) * factorial(n-i));
	else
		b = 0;
	end
end
