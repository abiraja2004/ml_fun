## Initialization
clear; close all; clc

## cost function
function J = computeCost(X, y, theta)
  m = length(y);
  h = X*theta;
  error = (h-y).^2;
  J = (1/(2*m))*sum(error);
end

## gradient descent
function [theta, J_history] = gradientDescent(X, y, theta, alpha, \
					      num_iters)
  m = length(y);
  J_history = zeros(num_iters, 1);
  J0 = computeCost(X, y, theta);
  for iter = 1:num_iters
    gradient = (alpha/m)*(X'*((X*theta)-y));
    theta = theta - gradient;
    J_new = computeCost(X, y, theta);
    J_history(iter) = J_new;
    if(J_new>J0),
      break,fprintf('[Error] Bad alpha: J_i > J_0');
    elseif(J0==J_new)
      break;
    end;
    if (iter > 1) && \
       (abs((J_new-J_history(iter-1))/J_history(iter-1)) < 1e-6)
      fprintf('[Status] Converged on iteration %i of %i\n', iter, \
	      num_iters);
      return;
    end
  end
end

## see the data
arg_list = argv ();
filename = arg_list{1};
data = dlmread(filename,',',0,0);
X = data(:, 1);
y = data(:, 2);
m = length(y);

## perform fit
X = [ones(m, 1), data(:,1)];
theta = zeros(2, 1);
iterations = 20000;
alpha = 0.01;
fprintf('[Status] Initial theta values: %f %f \n', theta(1), theta(2));

## compute and display initial cost
icost = computeCost(X, y, theta);
fprintf('[Status] Initial cost: %.6f \n', icost);

## run gradient descent
theta = gradientDescent(X, y, theta, alpha, iterations);
fcost = computeCost(X, y, theta);
fprintf('[Status] Final cost: %.6f\n', fcost);

## print theta to screen
fprintf('[Status] Final theta values: %f %f \n', theta(1), theta(2));
