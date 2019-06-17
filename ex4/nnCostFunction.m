function [J grad] = nnCostFunction(nn_params, ...
                                   input_layer_size, ...
                                   hidden_layer_size, ...
                                   num_labels, ...
                                   X, y, lambda)
%NNCOSTFUNCTION Implements the neural network cost function for a two layer
%neural network which performs classification
%   [J grad] = NNCOSTFUNCTON(nn_params, hidden_layer_size, num_labels, ...
%   X, y, lambda) computes the cost and gradient of the neural network. The
%   parameters for the neural network are "unrolled" into the vector
%   nn_params and need to be converted back into the weight matrices. 
% 
%   The returned parameter grad should be a "unrolled" vector of the
%   partial derivatives of the neural network.
%

% Reshape nn_params back into the parameters Theta1 and Theta2, the weight matrices
% for our 2 layer neural network
Theta1 = reshape(nn_params(1:hidden_layer_size * (input_layer_size + 1)), ...
                 hidden_layer_size, (input_layer_size + 1));

Theta2 = reshape(nn_params((1 + (hidden_layer_size * (input_layer_size + 1))):end), ...
                 num_labels, (hidden_layer_size + 1));

% Setup some useful variables
% m is no. of examples (in our case m=5000)
m = size(X, 1);


% ====================== YOUR CODE HERE ======================
% Instructions: You should complete the code by working through the
%               following parts.
%
% Part 1: Feedforward the neural network and return the cost in the
%         variable J. After implementing Part 1, you can verify that your
%         cost function computation is correct by verifying the cost
%         computed in ex4.m

% Add ones to the X data matrix -> for the intercept term
X = [ones(m, 1) X];

% create one-hot version of integer-labels vector y
I = eye(num_labels);
y_matrix = I(y,:);

% vectorized version
a1 = X;
z2 = a1 * Theta1';
% calculate the activation in layer 2
sigm_2 = sigmoid(z2);
% add bias term to get the full feature matrix
a2 = [ones(m,1) sigm_2];

z3 = a2 * Theta2';
a3 = sigmoid(z3);

hypo = a3;

% cost_matrix has dimension (5000x10)
cost_matrix = -y_matrix .* log(hypo) - (1-y_matrix) .* log(1-hypo);
cost = sum(sum(cost_matrix));

J_1 = cost/m;

%
% Part 2: Implement the backpropagation algorithm to compute the gradients
%         Theta1_grad and Theta2_grad. You should return the partial derivatives of
%         the cost function with respect to Theta1 and Theta2 in Theta1_grad and
%         Theta2_grad, respectively. After implementing Part 2, you can check
%         that your implementation is correct by running checkNNGradients
%
%         Note: The vector y passed into the function is a vector of labels
%               containing values from 1..K. You need to map this vector into a 
%               binary vector of 1's and 0's to be used with the neural network
%               cost function.
%
%         Hint: We recommend implementing backpropagation using a for-loop
%               over the training examples if you are implementing it for the 
%               first time.
%


% fetch parameters unrelated to bias b in layer 1 and 2 (1st column)
% remove first column Theta values -> belong to bias 
Theta1_reg = Theta1(:,2:end);
Theta2_reg = Theta2(:,2:end);
% squeeze and concatenate all params
params_reg = [Theta1_reg(:);Theta2_reg(:)];
% square parameters
params_reg_sq = params_reg.^2;

J_2 = (lambda * sum(params_reg_sq))/(2*m);

% add J_1 (fitting J term) and J_2 (regularization J term)

J = J_1 + J_2;


% Part 3: Implement regularization with the cost function and gradients.
%
%         Hint: You can implement this around the code for
%               backpropagation. That is, you can compute the gradients for
%               the regularization separately and then add them to Theta1_grad
%               and Theta2_grad from Part 2.
%

d3 = a3 - y_matrix; % d3 is (m x K) (here 5000 x 10)
d2 = d3 * Theta2(:,2:end) .* sigmoidGradient(z2); % is d2 is (m x 25)


% NOTE: Excluding the first column of Theta2 is because the hidden 
% layer bias unit has no connection to the input layer - so we do 
% not use backpropagation for it.

% Delta1 must have same dimensions as Theta1
Delta1 = d2'*a1;

% Delta1 must have same dimensions as Theta1
Delta2 = d3'*a2;

% fetch Theta1 and Theta2 and set the first column to 0 because we do not
% regularize the gradient of the bias-weights

Theta1_grad = Delta1/m;
Theta2_grad = Delta2/m;


% fetch Theta1 and Theta2 and set the first column to 0 because we do not
% regularize the gradient of the bias-weights

Theta1(:,1) = 0;
Theta2(:,1) = 0;

% determine the regularized Delta matrices
Theta1_grad = Theta1_grad + (lambda * Theta1)/m;
Theta2_grad = Theta2_grad + (lambda * Theta2)/m;

% -------------------------------------------------------------

% =========================================================================


% Unroll gradients
grad = [Theta1_grad(:) ; Theta2_grad(:)];

end


% WHAT FOLLOWS IS THE UNVECTORIZED VERSION OF COST CALCULATION

% for t=1:m % use t instead of i because i is denoting imaginary unit i for complex numbers :) 
%     % fetch example x_t
%     x_t = X(t,:)';
%     a1 = x_t;
%     % transpose because hypo is (10x1)
%     y_t = y_matrix(t,:)';
%     
%     % now calculate hypothesis h(x_i)
%     z2 = Theta1 * a1;
%     sigm_1 = sigmoid(z2); % a_1 is (25x1)
%     % add bias term
%     a2 = [1;sigm_1]; % a_1 is (26x1) -> the added bias is +1 and NOT 0!
%     
%     z3 = Theta2 * a2; % z_2 is (10x1)
%     sigm_2 = sigmoid(z3);
%     a3 = sigm_2;
%     hypo = a3;
%     
%     % calcuate cost(t) for each example x_t
%     cost_t_vec = -y_t .* log(hypo) - (1-y_t) .* log(1-hypo);
%     % summation over all 10 errors
%     cost_t = sum(cost_t_vec);
%     J_1 = cost_t;
%     
% end
% 
% % divide total cost by no. of examples (m)
% J_1 = J_1*(1/m);
