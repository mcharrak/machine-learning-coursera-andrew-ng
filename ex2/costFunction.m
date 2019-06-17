function [J, grad] = costFunction(theta, X, y)
%COSTFUNCTION Compute cost and gradient for logistic regression
%   J = COSTFUNCTION(theta, X, y) computes the cost of using theta as the
%   parameter for logistic regression and the gradient of the cost
%   w.r.t. to the parameters.

% Initialize some useful values
m = length(y); % number of training examples

% You need to return the following variables correctly 

% calculate the dot-product of X and theta
dot = X*theta;
%fprintf('size of dot-product: %s\n', mat2str(size(dot)));

% calculate the sigmoid output of dot-product
hypo = sigmoid(dot);
%fprintf('size of hypothesis: %s\n', mat2str(size(hypo)));

% define cost function J
J = (-y' * log(hypo) - (1-y)' * log(1-hypo))/m;

%fprintf('size of cost function J: %s\n', mat2str(size(J)));

% define gradients grad
grad = (1/m) * X' * (hypo - y);


% ====================== YOUR CODE HERE ======================
% Instructions: Compute the cost of a particular choice of theta.
%               You should set J to the cost.
%               Compute the partial derivatives and set grad to the partial
%               derivatives of the cost w.r.t. each parameter in theta
%
% Note: grad should have the same dimensions as theta
%








% =============================================================

end
