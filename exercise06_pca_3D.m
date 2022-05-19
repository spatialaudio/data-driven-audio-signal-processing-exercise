% Sascha Spors, Professorship Signal Theory and Digital Signal Processing,
% Institute of Communications Engineering (INT), Faculty of Computer Science
% and Electrical Engineering (IEF), University of Rostock, Germany
%
% Data Driven Audio Signal Processing - A Tutorial with Computational Examples
% Feel free to contact lecturer frank.schultz@uni-rostock.de
%
% Exercise 06: SVD / PCA on 3D data

clear all
%close all
clc

rng(4)  % seed
xmax = 6;

N = 200;
R = 3;
mu = [0, 0, 0];
sigma = [3, 2, 0; 2, 3, 1; 0, 1, 1];
A = mvnrnd(mu, sigma, N);
A = A - mean(A, 1);  % if mean free columns, the SVD and pca() are the same
if R ~= rank(A)  % check that we have desired rank
    disp('!!! matrix A has not the desired rank R !!!')
end

[coeff, score, latent] = pca(A);
[U, S, V] = svd(A);
disp('A V = U S')
allclose(A*V, U*S)

% principal component signals / principal component scores
pcs = U*S;
% principal component loadings / principal component coefficients
pcl = V;

%% check if pca() == SVD solution numerically
disp('check if SVD-based PCA == pca()')
% both results should only differ in polarity
% so we first get a polarity from (1,1)-entry
polarity = pcs(1,1) / score(1,1);
% and check by allclose
allclose(score, polarity*pcs) & allclose(coeff, polarity*pcl)

%% check 4 subspaces of A
disp('4 subspace check of A -> orth() / null() vs. SVD data')
allclose(orth(A), U(:, 1:R)) % check column space == 1st R cols of U
null(A)  % check null space == zero vec
allclose(orth(A'), V)  % check row space == all V
allclose(null(A'), U(:, R+1:end))  % check left null space, remaining cols of U->large

%%
disp('the variance of the principal component signals/scores')
var(pcs).'
disp('is equal to the eigen values of (A''*A) / (N-1)')
RA = A'*A / (size(A, 1)-1);
sort(eig(RA), 'descend')
disp('is equal to the (singular values of A)^2 / (N-1)')
diag(S'*S) / (size(A, 1)-1)
disp('is equal to latent coeff from pca()')
latent

%%
R_red = 2;
Dim_red = R_red;

% rank1-matrix superposition up to rank R_red
S_tmp = S;
for r = R_red+1:size(S, 2)
    S_tmp(r, r) = 0;
end
A_R_red = (U*S_tmp) * V';
% we can do this also by setting corresponding PC loadings to zero
V_red = V;
V_red(:, R_red+1:end) = 0;
disp('rank1-superposition up to R_red == (U S) * V_red')
allclose((U*S) * V_red', A_R_red)

A_Dim_red = A*0;  % we use same dim as A for convenient 3D plot of the data
% however the last (R - Dim_red) columns are exactly zero and thus the matrix
% is actually of dim N x Dim_red
A_Dim_red(:, 1:Dim_red) = pcs(:, 1:Dim_red);  % pcs = U*S
disp('cf. the lecture approach, slide 5-23:')
disp('pcs(:, 1:Dim_red) == A * V(:, 1:Dim_red)')
allclose(A_Dim_red(:, 1:Dim_red), A * V(:, 1:Dim_red))

% if Dim_red == R_red the matrix is same as A_R_red from above
disp('A V = U S -> A V_red V_red'' = U S V_red'' for reduced number of vectors in V')
disp('is same as rank1-matrix superposition up to rank R_red')
allclose((A * V(:, 1:Dim_red)) * V(:, 1:Dim_red)', A_R_red)
% in lecture known as
% 'Reconstruction using a limited number of principle components', cf. slide 5-23
% which in fact is a rank reduction of A by using only Dim_red == R_red
% largest singular values

%% plot
ax_original_feature_space = subplot(1, 4, 1);
for n = 1:N
    plot3(A(n, 1), A(n, 2), A(n, 3), 'ko'), hold on
end
plot3([0, 5*V(1,1)],[0, 5*V(2,1)],[0, 5*V(3,1)], 'color', '#1f77b4', 'linewidth', 6)
plot3([0, 5*V(1,2)],[0, 5*V(2,2)],[0, 5*V(3,2)], 'color', '#ff7f0e', 'linewidth', 4)
plot3([0, 5*V(1,3)],[0, 5*V(2,3)],[0, 5*V(3,3)], 'color', '#2ca02c', 'linewidth', 2)
hold off
xlim([-xmax, xmax])
ylim([-xmax, xmax])
zlim([-xmax, xmax])
axis square
xlabel('Original Feature 1 A[:,1]')
ylabel('Original Feature 2 A[:,2]')
zlabel('Original Feature 3 A[:,3]')
title('Original Data A')
grid on

ax_pc_space = subplot(1, 4, 2);
for n = 1:N
    plot3(pcs(n, 1), pcs(n, 2), pcs(n, 3), 'ko'), hold on
end
% we can plot PC directions
% make sure that the example has no reflection in the SVD data!
% hence usage of seed rng(4)
plot3([0, 5], [0, 0], [0, 0], 'color', '#1f77b4', 'linewidth', 6)
plot3([0, 0], [0, 5], [0, 0], 'color', '#ff7f0e', 'linewidth', 4)
plot3([0, 0], [0, 0], [0, 5], 'color', '#2ca02c', 'linewidth', 2)
hold off
xlim([-xmax, xmax])
ylim([-xmax, xmax])
zlim([-xmax, xmax])
axis square
xlabel('\sigma_1 U[:,1] = A V[:,1]')
ylabel('\sigma_2 U[:,2] = A V[:,2]')
zlabel('\sigma_3 U[:,3] = A V[:,3]')
title('Data in Weighted U Space == Data Projected onto V')
grid on

subplot(1, 4, 3)
for n = 1:N
    plot3(A_R_red(n, 1), A_R_red(n, 2), A_R_red(n, 3), 'ko'), hold on
end
hold off
xlim([-xmax, xmax])
ylim([-xmax, xmax])
zlim([-xmax, xmax])
axis square
xlabel('New Feature 1')
ylabel('New Feature 2')
zlabel('New Feature 3')
title('A Reduced to Rank 2, i.e. Plane in 3D space')
grid on

% A_Dim_red is actually only 2D, but to see this fact in a 3D plot
% we use this plot3 handling
subplot(1, 4, 4)
for n = 1:N
    plot3(A_Dim_red(n, 1), A_Dim_red(n, 2), A_Dim_red(n, 3), 'ko'), hold on
end
hold off
xlim([-xmax, xmax])
ylim([-xmax, xmax])
zlim([-xmax, xmax])
axis square
xlabel('\sigma_1 U[:,1]')
ylabel('\sigma_2 U[:,2]')
zlabel('\sigma_3 U[:, 3] \cdot 0')
title('U/V Space Dimensionality Reduced, i.e. Plane in 2D space')
grid on
view([0 90])


%%
disp('percentage of explained variance')
latent/sum(latent) * 100
disp('cumsum percentage of explained variance')
cumsum(latent)/sum(latent)*100

disp('xmax for axis lim larger than occuring values')
max([max(max(abs(A)))...
    max(max(abs(pcs)))...
    max(max(abs(A_R_red)))...
    max(max(abs(A_Dim_red)))]) < xmax

%%
% check specific data points
n1 = 187;
n2 = 120;
n3 = 60;

axes(ax_original_feature_space)
hold on
plot3(A(n1,1), A(n1,2), A(n1,3), '*', 'color', '#d62728', 'markersize', 10)
plot3(A(n2,1), A(n2,2), A(n2,3), 'p', 'color', '#e377c2', 'markersize', 10)
plot3(A(n3,1), A(n3,2), A(n3,3), 'h', 'color', '#17becf', 'markersize', 12)
hold off
axes(ax_pc_space)
hold on
plot3(S(1,1) * U(n1,1), S(2,2) * U(n1,2), S(3,3) * U(n1,3), '*', 'color', '#d62728', 'markersize', 10)
plot3(S(1,1) * U(n2,1), S(2,2) * U(n2,2), S(3,3) * U(n2,3), 'p', 'color', '#e377c2', 'markersize', 10)
plot3(S(1,1) * U(n3,1), S(2,2) * U(n3,2), S(3,3) * U(n3,3), 'h', 'color', '#17becf', 'markersize', 12)
hold off


%##############################################################################
function flag = allclose(a, b)
% https://numpy.org/doc/stable/reference/generated/numpy.allclose.html
% numpy.allclose(a, b, rtol=1e-05, atol=1e-08, equal_nan=False)
% https://stackoverflow.com/questions/28975822/matlab-equivalent-for-numpy-allclose
rtol=1e-05;
atol=1e-08;
flag = all( abs(a(:)-b(:)) <= atol+rtol*abs(b(:)) );
end
