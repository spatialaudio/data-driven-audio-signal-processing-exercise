% Sascha Spors, Professorship Signal Theory and Digital Signal Processing,
% Institute of Communications Engineering (INT), Faculty of Computer Science
% and Electrical Engineering (IEF), University of Rostock, Germany
%
% Data Driven Audio Signal Processing - A Tutorial with Computational Examples
% Feel free to contact lecturer frank.schultz@uni-rostock.de
%
% Ex10: XOR with linear/nonlinear model
clear all
%close all
clc
%%
disp('xor with a linear model')
% try XOR as y = b*X(:,1) + w1*X(:,2) + w2*X(:,3)
X = [0 0; 0 1; 1 0; 1 1]
y = [0; 1; 1; 0]

% train model by least squares / left inverse (tall/thin, full col rank)
Xtilde = [ones(4,1) X];
wtilde = inv(Xtilde.'*Xtilde)*Xtilde.' * y;

b = wtilde(1)  % bias
w = wtilde(2:3)  % weights w1 and w2

% model output -> 0.5 everywhere, xor cannot be realized with this model
yhat = Xtilde*wtilde
% we can figure why by discussing left/right null spaces, row/col space of
% Xtilde and/or we can check the U,V matrices of the SVD
%[U,S,V] = svd(Xtilde) to see that this cannot work
%[U,S,V] = svd(Xtilde);
%tmp = pinv(V) * wtilde;
%Xtilde*(tmp(1)*V(:,1) + tmp(2)*V(:,2) + tmp(3)*V(:,3))


%%
disp('xor with a ReLU nonlinear model')
% let's transpose data do be consistent with Andrew Ng's coding convention:
% see course https://www.coursera.org/learn/neural-networks-deep-learning
% especially week 2
X = [0 0; 0 1; 1 0; 1 1].'
y = [0; 1; 1; 0].'

% we could learn a model, we really should try this ourselves,
% however the result for a simple two layer model is known from
% I. Goodfellow, Y. Bengio, A. Courville, Deep Learning. MIT Press, 2016, ch 6.1
w_inner = [1 1; 1 1];
b_inner = [0; -1];
w_outer = [1; -2];

yhat = w_outer.' * max(w_inner.'*X + b_inner, 0)
% step by step:
tmp = w_inner.'*X + b_inner;  % input layer perceptron
tmp = max(tmp, 0);  % nonlinear activation ReLU
tmp = w_outer.'*tmp;  %output layer perceptron (no bias, linear activation)

%%
% Copyright
% - the script is provided as [Open Educational Resources](https://en.wikipedia.org/wiki/Open_educational_resources)
% - comment text is licensed under [Creative Commons Attribution 4.0](https://creativecommons.org/licenses/by/4.0/)
% - Matlab code is licensed under the [MIT license](https://opensource.org/licenses/MIT)
% - feel free to use for your own purposes
% - please attribute the work as follows:
% Frank Schultz, Data Driven Audio Signal Processing-A Tutorial Featuring
% Computational Examples, University of Rostock* ideally with relevant
% file(s), github URL
% https://github.com/spatialaudio/data-driven-audio-signal-processing-exercise,
% commit number and/or version tag, year.
