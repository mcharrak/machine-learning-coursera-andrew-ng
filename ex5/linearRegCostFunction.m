function [J, grad] = linearRegCostFunction(X, y, theta, lambda)
%LINEARREGCOSTFUNCTION Compute cost and gradient for regularized linear 
%regression with multiple variables
%   [J, grad] = LINEARREGCOSTFUNCTION(X, y, theta, lambda) computes the 
%   cost of using theta as the parameter for linear regression to fit the 
%   data points in X and y. Returns the cost in J and the gradient in grad

% Initialize some useful values
m = length(y); % number of training examples

% You need to return the following variables correctly 
J = 0;
grad = zeros(size(theta));

% ====================== YOUR CODE HERE ======================
% Instructions: Compute the cost and gradient of regularized linear 
%               regression for a particular choice of theta.
%
%               You should set J to the cost and grad to the gradient.
%

% Add ones to the X data matrix -> for the intercept term
% X = [ones(m, 1) X]; %no need to because this is already done in the
% function call within the file ex5.mlx :)

% theta is 2x1 and single x is mx2 cost must be mx1

% get hypothesis which must be in vectorized form mx1 (here same as size(y))
hypo = X*theta;

diff = hypo - y;
diff_sq = (diff).^2;
J_1 = sum(diff_sq)/(2*m);

%get sum of regularization relevant squared params
sum_theta_sq = sum(theta(2:end).^2);

J_2 = (lambda * sum_theta_sq)/(2*m);
% add loss terms to get full cost
J = J_1 + J_2;

% calculate gradient for intercept term theta_0 only using 
% the first column of our design matrix X, which is all ones anyway :)
% grad_0 is a scalar because it belongs to the bias weight i.e. theta_0
grad_0 = (1/m) * X(:,1)' * (hypo - y);

% calculate gradients for all other theta parameters using columns (2:end)
% of our design matrix X

% X(:,1) is of dimension (mx1)
% X(:,2:end) is of dimension (mxn)
% (hypo-y) is of dimension (mx1)
grad_1 = (1/m) * X(:,2:end)' * (hypo - y)  + (lambda/m) * theta(2:end);

% stack both gradients for j=0 and all others for j=1 to n in total 
% n+1 partial derivatives
grad = [grad_0;grad_1];

% =========================================================================

grad = grad(:);

end
