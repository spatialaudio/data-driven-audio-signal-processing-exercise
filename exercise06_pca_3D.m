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

rng(4)
xmax = 6;

N = 200;
R = 3;
mu = [0, 0, 0];
sigma = [3, 2, 0; 2, 3, 1; 0, 1, 1];
A = mvnrnd(mu, sigma, N);
A = A - mean(A);  % if mean free col SVD and pca() are same
if R ~= rank(A)  % check rank
    disp('!!! matrix A has not the desired rank R !!!')
end

[coeff, score, latent] = pca(A);
[U, S, V] = svd(A);
pcs = U*S;  % known as principal component signals/scores
pcl = V;  % known as principal component loadings/coefficients

%% check pca() vs. SVD solution -> must yield numerical zero
tmp = 0;
for c=1:size(score, 2)
    tmp = tmp + min(norm(pcs(:, c) + score(:, c)), norm(pcs(:, c) - score(:, c)));
    tmp = tmp + min(norm(pcl(:, c) + coeff(:, c)), norm(pcl(:, c) - coeff(:, c))); 
end
disp('we should get numerical zero which indicates our manual SVD == pca()')
tmp / size(score, 2) / 2

%% check 4 subspaces
disp('4 subspace check of A -> orth() / null() vs. SVD data')
norm(orth(A) - U(:, 1:R),'fro')  % check column space == 1st two cols of U
null(A)  % check null space == zero vec
norm(orth(A') - V, 'fro')  % check row space == all V
norm(null(A') - U(:, R+1:end), 'fro')  % check left null space, remaining cols of U->large

%%
disp('the variance of the principal component signals')
var(pcs).'
disp('is equal to the eigvals(A''*A) / (N-1)')
R = A'*A / (size(A, 1)-1);
sort(eig(R), 'descend')
disp('is equal to the (sing vals of A)^2 / (N-1)')
diag(S'*S) / (size(A, 1)-1)
disp('is equal to latent from pca()')
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
% V_red = V;
% V_red(:, R_red+1:end) = 0;
% A_R_red = (U*S) * V_red';

A_Dim_red = A*0;  % we use same dim as A for convenient plotting the data
% however the last R-Dim_red columns are exactly zero and thus the matrix
% is of dim N x Dim_red
A_Dim_red(:, 1:Dim_red) = pcs(:, 1:Dim_red);  % pcs = U*S
% cf. the lecture approach, slide 5-23:
%   tmp = A*0;
%   tmp(:, 1:Dim_red) = A * V(:, 1:Dim_red);

% if Dim_red == R_red the matrix
A_R_red2 = A_Dim_red(:, 1:Dim_red) * V(:, 1:Dim_red)';
% is same as A_R_red from above
norm(A_R_red2-A_R_red,'fro')
% in lecture known as
% Reconstruction using a limited number of principle components, cf. slide 5-23

%% plot
subplot(1, 4, 1)
for n = 1:N
    plot3(A(n, 1), A(n, 2), A(n, 3), 'ko'), hold on 
end
hold off
xlim([-xmax, xmax])
ylim([-xmax, xmax])
zlim([-xmax, xmax])
axis square
xlabel('Feature 1')
ylabel('Feature 2')
zlabel('Feature 3')
title('original data')
grid on

subplot(1, 4, 2)
for n = 1:N
    plot3(pcs(n, 1), pcs(n, 2), pcs(n, 3), 'ko'), hold on 
end
hold off
xlim([-xmax, xmax])
ylim([-xmax, xmax])
zlim([-xmax, xmax])
axis square
xlabel('\sigma_0 U[0]')
ylabel('\sigma_1 U[1]')
zlabel('\sigma_3 U[2]')
title('data in U space')
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
title('rank reduced')
grid on

subplot(1, 4, 4)
for n = 1:N
    plot3(A_Dim_red(n, 1), A_Dim_red(n, 2), A_Dim_red(n, 3), 'ko'), hold on 
end
hold off
xlim([-xmax, xmax])
ylim([-xmax, xmax])
zlim([-xmax, xmax])
axis square
xlabel('\sigma_0 U[0]')
ylabel('\sigma_1 U[1]')
zlabel('\sigma_3 U[2]')
title('U space dimensionality reduced')
grid on



%%
disp('percentage of explained variance')
latent/sum(latent) * 100
disp('cumsum percentage of explained variance')
cumsum(latent)/sum(latent)*100

xmax
disp('xmax should be larger than')
max([max(max(abs(A)))...
max(max(abs(pcs)))...
max(max(abs(A_R_red)))...
max(max(abs(A_Dim_red)))])








