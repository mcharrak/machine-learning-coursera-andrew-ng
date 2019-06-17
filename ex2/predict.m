function p = predict(theta, X)
%PREDICT Predict whether the label is 0 or 1 using learned logistic 
%regression parameters theta
%   p = PREDICT(theta, X) computes the predictions for X using a 
%   threshold at 0.5 (i.e., if sigmoid(theta'*x) >= 0.5, predict 1)

m = size(X, 1); % Number of training examples

%get probabilites
probs = sigmoid(X*theta);

% creat prediction vector with all values zero
p = zeros(m,1);

% find indices where probabilites larger 0.5
pos_ind = probs >=0.5;

% set values to 1 (at indices where prob larger 0.5)
p(pos_ind) = 1;

% ====================== YOUR CODE HERE ======================
% Instructions: Complete the following code to make predictions using
%               your learned logistic regression parameters. 
%               You should set p to a vector of 0's and 1's
%







% =========================================================================


end
