## Initialization
clear; close all; clc

## cost function
function J = computeCost(X, y, theta)
  m = length(y);
  J = 0;
end

## gradient descent
function [theta, J_history] = gradientDescent(X, y, theta, \
					      alpha, num_iters)
  m = length(y);
  J_history = zeros(num_iters, 1);
  J_0 = computeCost(X, y, theta);
  for iter = 1:num_iters
    J_history(iter) = computeCost(X, y, theta);
  end
end

## normalization function
function [X_norm, mu, sigma] = featureNormalize(X)
  X_norm = X;
  mu = zeros(1, size(X, 2));
  sigma = zeros(1, size(X, 2));
  mu = mean(X);
  sigma = std(X);
  for i=1:length(X)
    X_norm(i,:) = (X(i,:)-mu)./sigma;
  end
end

## closed-form solution to linreg using normal eqn
function [theta] = normalEqn(X, y)
  theta = zeros(size(X, 2), 1);
  theta = inverse(X'*X)*X'*y;
end

## see the data
arg_list = argv ();
filename = arg_list{1};
data = dlmread(filename,',',0,0);
X = data(:, 1:2);
y = data(:, 3);
m = length(y);

## Scale features and set them to zero mean
[X mu sigma] = featureNormalize(X);

## Add intercept term to X
X = [ones(m, 1) X];

## Choose some alpha value
alpha = 0.01;
num_iters = 400;

## Init Theta and Run Gradient Descent 
theta = zeros(3, 1);
[theta, J_history] = gradientDescent(X, y, theta, alpha, num_iters);

% Estimate the price of a 1650 sq-ft, 3 br house
% ====================== YOUR CODE HERE ======================
% Recall that the first column of X is all-ones. Thus, it does
% not need to be normalized.
price = 0; % You should change this


% ============================================================

fprintf(['Predicted price of a 1650 sq-ft, 3 br house ' ...
         '(using gradient descent):\n $%f\n'], price);

fprintf('Program paused. Press enter to continue.\n');
pause;

%% ================ Part 3: Normal Equations ================

fprintf('Solving with normal equations...\n');

% ====================== YOUR CODE HERE ======================
% Instructions: The following code computes the closed form 
%               solution for linear regression using the normal
%               equations. You should complete the code in 
%               normalEqn.m
%
%               After doing so, you should complete this code 
%               to predict the price of a 1650 sq-ft, 3 br house.
%

%% Load Data
data = csvread('ex1data2.txt');
X = data(:, 1:2);
y = data(:, 3);
m = length(y);

% Add intercept term to X
X = [ones(m, 1) X];

% Calculate the parameters from the normal equation
theta = normalEqn(X, y);

% Display normal equation's result
fprintf('Theta computed from the normal equations: \n');
fprintf(' %f \n', theta);
fprintf('\n');


% Estimate the price of a 1650 sq-ft, 3 br house
% ====================== YOUR CODE HERE ======================
price = 0; % You should change this


% ============================================================

fprintf(['Predicted price of a 1650 sq-ft, 3 br house ' ...
         '(using normal equations):\n $%f\n'], price);

