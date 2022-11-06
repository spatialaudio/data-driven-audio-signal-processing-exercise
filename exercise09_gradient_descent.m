% Sascha Spors, Professorship Signal Theory and Digital Signal Processing,
% Institute of Communications Engineering (INT), Faculty of Computer Science
% and Electrical Engineering (IEF), University of Rostock, Germany
%
% Data Driven Audio Signal Processing - A Tutorial with Computational Examples
% Feel free to contact lecturer frank.schultz@uni-rostock.de
%
% Exercise 09: Gradient Descent

clear all
%close all
clc

plot3_flag = 'surface';
%plot3_flag = 'contour';

syms x1 x2 % weights in the model
x = [x1; x2]; % set up a vector

% create a bowl with min at 4,-1 and fmin=-5
S = [1 1; 1 2];
a = [3; 2];

% create a valley with min at 0,0 and fmin=0
%S = [1 0; 0 1/100];
%a = [0;0];

S_is_posdef = all(eig(S)) > 0  % must be pos (semi)-def
disp('inv(S)')
inv(S)
f = simplify(1/2*x.'*S*x - a.'* x)  % set up f(x1,x2)

% analytical solutions
% cf. textbook Gilbert Strang (2019):
% "Linear Algebra and Learning from Data", Wellesley, Ch. VI.4
xmin = inv(S)*a
fmin = -1/2 * a.' * xmin

gradf = [diff(f, x1); diff(f, x2)] % in DNN learning the loss function is
% chosen such that gradient is known analytically and can be evaluated
% numerically at specific points, this is one of the jobs of backpropagation

% random init the weights
xk = [-4; -3];
% random init momentum vector
zk = [0;0];

% settings for gradient descent / hyper parameters
gd_flag = 'normal_gd'
%gd_flag = 'momentum_gd'
momentum_coeff = 1/3;


%step_size = 0.01; % with 'normal_gd': very slow, not reaching min with chosen steps
%or
%step_size = 0.2; % with 'normal_gd': smooth descent
%or
step_size = 0.5; % with 'normal_gd': zig zag at start, then smooth
%or
%step_size = 0.7; % with 'normal_gd': zig zag, but meets min, very dangerous to use
%or
%step_size = 0.8; % with 'normal_gd':  zig zag, useless, because ascent
%check all step_sizes with gd_flag = 'momentum_gd'


steps = 100;

%%
plot_data = zeros(3, steps);
if strcmp(plot3_flag, 'surface')
    fsurf(f, [-6 6 -6 6])
elseif strcmp(plot3_flag, 'contour')
    fcontour(f, [-6 6 -6 6], 'LevelStep', 2)
else
    disp('check plot3_flag')
end
hold on % to add gradient descent points
for k = 1:steps
    % get data points on the surface f(x1,x2) following the gradient descent
    % for plot3d below
    tmp = f;
    tmp = subs(tmp, x1, xk(1));
    farg = subs(tmp, x2, xk(2));
    plot_data(1, k) = xk(1);
    plot_data(2, k) = xk(2);
    plot_data(3, k) = farg;
    
    if strcmp(gd_flag, 'normal_gd') % calc gradient descent
        tmp = gradf; % tmp var
        tmp = subs(tmp, x1, xk(1)); %set current x1 value
        tmp = subs(tmp, x2, xk(2)); %set current x2 value
        xk = xk - step_size * tmp; % update rule for gradient descent xk -> xk+1
    elseif strcmp(gd_flag, 'momentum_gd') % calc gradient descent with momentum
        % cf. textbook Gilbert Strang (2019):
        % "Linear Algebra and Learning from Data", Wellesley, p.351
        xk = xk - step_size * zk; % xk -> xk+1
        
        % grad for xk+1:
        tmp = gradf; % tmp var
        tmp = subs(tmp, x1, xk(1)); %set current x1 value
        tmp = subs(tmp, x2, xk(2)); %set current x2 value
        
        % update zk -> zk+1
        zk = momentum_coeff * zk + tmp;
    else
        disp('check gd_flag')
    end
end
plot3(plot_data(1,:), plot_data(2,:), plot_data(3,:), 'ro-')
plot3(plot_data(1,1), plot_data(2,1), plot_data(3,1), 'ro', 'markersize', 10)
plot3(plot_data(1,end), plot_data(2,end), plot_data(3,end), 'rx', 'markersize', 10)
plot3(xmin(1), xmin(2), fmin, 'ok', 'markersize', 10)
hold off
xlim([-6 6])
ylim([-6 6])
axis square
view([0 90])
xlabel('weight x_1')
ylabel('weight x_2')
title(eval(xk))
disp('numerical min vs. exact min')
eval(xk)
xmin

%% Newton method
disp('Newton method')
% clc
H = [diff(gradf,x1) diff(gradf,x2)] % == for quadratic problems H=S
S
% since H=S has no dependency on x1 and x2, Newton solves the problem in just
% one step (we should not expect that this happens in practical applications)
% so this 'gold' standard is of theoretical interest
xk = [-4; -3]; % random init point
tmp = gradf;
tmp = subs(tmp, x1, xk(1)); %set current x1 value
tmp = subs(tmp, x2, xk(2)); %set current x2 value
% cf. textbook Gilbert Strang (2019):
% "Linear Algebra and Learning from Data", Wellesley, p.327 (3),(4)
% solve with Newton in one step
inv(S) * (S*xk - tmp); % or rewritten
xk = eval(xk - inv(S)*tmp)

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
