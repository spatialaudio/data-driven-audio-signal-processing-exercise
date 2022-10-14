% Sascha Spors, Professorship Signal Theory and Digital Signal Processing,
% Institute of Communications Engineering (INT), Faculty of Computer Science
% and Electrical Engineering (IEF), University of Rostock, Germany
%
% Data Driven Audio Signal Processing - A Tutorial with Computational Examples
% Feel free to contact lecturer frank.schultz@uni-rostock.de
%
% MIT License for this code snippet applies
%
% SVD and the 4 Subspaces - A Number Example
% we might consider the matrix A from slide 4/30 of the highly recommended
% web resource "A 2020 Vision of Linear Algebra, Gilbert Strang, MIT, 2020"
% see https://ocw.mit.edu/courses/res-18-010-a-2020-vision-of-linear-algebra-spring-2020/resources/mitres_18_010s20_la_slides/

clear all
close all
clc

X = [1 4 5;
     3 2 5;
     2 1 3];

[U,S,V]=svd(X);
disp('rank r = ')
r = rank(X)
%%
disp('nullspace of X, which is a line (1D) in 3D space')
null(X)
V(:,r+1:end)
V(:,r+1:end)/0.577350269189626
%%
disp('left nullspace of X, which  is a line (1D) in 3D space')
null(X')
U(:,r+1:end)
U(:,r+1:end)/-0.0816496580927726
disp('column space of X, which is a plane (2D) in 3D space')
U(:,1:r)
disp('row space space of X, which is a plane (2D) in 3D space')
V(:,1:r)
%%
disp('linear combinations of U to get original columns of X')
w1 = inv(U) * [1; 3; 2]  % actually we could avoid inv, in favor for U',
% because for orthonormal / unitary matrices U'=U^-1 holds,
% but to make a little clearer that we solve an inverse problem here
% we use go for inv
% later on, we will reveal the nice projection properties of U and V, where
% U' is much more meaningful
U * w1

w2 = inv(U) * [4; 2; 1]
U * w2

w3 = inv(U) * [5; 5; 3]
U * w3
% w1+w2 == w3 % must hold because col3 is lin dep of col1 and col2, might
% be numerically not precise
%%
disp('linear combinations of V to get original rows of X (i.e. original columns of X^T)')
w1 = inv(V) * [1;4;5]
(V*w1)' % output as a row

w2 = inv(V) * [3;2;5]
(V*w2)' % output as a row

w3 = inv(V) * [2;1;3]
(V*w3)' % output as a row
%%
disp('span the row space by two other vectors')
% on slide 5/30 actually two simple numbered vectors that span the row space
% are given. These can be derived by the nice X = C R factorization (we might
% want to check this out as well)
% so instead taking V(:,1:r) we can use the (not unit length!, not orthogonal!)
RS = [0 1; 1 0; 1 1]
% with these row space vectors we can find nice numbered weights to get the
% original rows of X
w1 = pinv(RS) * [1;4;5]
(RS*w1)' % output as a row

w2 = pinv(RS) * [3;2;5]
(RS*w2)' % output as a row

w3 = pinv(RS) * [2;1;3]
(RS*w3)' % output as a row
 