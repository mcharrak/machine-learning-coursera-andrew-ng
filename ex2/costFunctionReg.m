function [J, grad] = costFunctionReg(theta, X, y, lambda)
%COSTFUNCTIONREG Compute cost and gradient for logistic regression with regularization
%   J = COSTFUNCTIONREG(theta, X, y, lambda) computes the cost of using
%   theta as the parameter for regularized logistic regression and the
%   gradient of the cost w.r.t. to the parameters. 

% Initialize some useful values
m = length(y); % number of training examples

% You need to return the following variables correctly 

% calculate the dot-product of X and theta
dot = X*theta;
%fprintf('size of dot-product: %s\n', mat2str(size(dot)));

% calculate the sigmoid output of dot-product
hypo = sigmoid(dot);
%fprintf('size of hypothesis: %s\n', mat2str(size(hypo)));

% define cost function J with regularization term

% J_1 is reconstruction term
J_1 = (-y' * log(hypo) - (1-y)' * log(1-hypo))/m;

% J_2 is regularization term
J_2 = 0.5* (lambda/m) * sum(theta(2:end).^2);

% total cost function is reconstruction + regularization term
J = J_1 + J_2;

% calculate gradient for intercept term theta_0 only using the first column
% of our design matrix X
grad_0 = (1/m) * X(:,1)' * (hypo - y);

% calculate gradients for all other theta parameters using columns (2:end)
% of our design matrix X
grad_1 = (1/m) * X(:,2:end)' * (hypo - y)  + (lambda/m)*theta(2:end);

% stack both gradients for j=0 and all others for j=1 to n in total 
% n+1 partial derivatives
grad = [grad_0;grad_1];

% ====================== YOUR CODE HERE ======================
% Instructions: Compute the cost of a particular choice of theta.
%               You should set J to the cost.
%               Compute the partial derivatives and set grad to the partial
%               derivatives of the cost w.r.t. each parameter in theta






% =============================================================

end
