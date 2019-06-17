function p = predict(Theta1, Theta2, X)
%PREDICT Predict the label of an input given a trained neural network
%   p = PREDICT(Theta1, Theta2, X) outputs the predicted label of X given the
%   trained weights of a neural network (Theta1, Theta2)

% Useful values
m = size(X, 1);
num_labels = size(Theta2, 1);

% You need to return the following variables correctly

% Add ones to the X data matrix
X = [ones(m, 1) X];

% calculate activations of 2nd layer
% a_2 and z_2 are of dimension: 5000 x 25
z_2 = X * Theta1';
a_2 = sigmoid(z_2);

% Add ones to the a_2 data matrix
a_2 = [ones(m, 1) a_2];

% calculate activations of 3rd layer
% a_2 of dimension 5000 x 26
% a_3 and z_3 are of dimension: 5000 x 10
z_3 = a_2 * Theta2';
a_3 = sigmoid(z_3);

[max_values, p] = max(a_3,[],2);


% ====================== YOUR CODE HERE ======================
% Instructions: Complete the following code to make predictions using
%               your learned neural network. You should set p to a 
%               vector containing labels between 1 to num_labels.
%
% Hint: The max function might come in useful. In particular, the max
%       function can also return the index of the max element, for more
%       information see 'help max'. If your examples are in rows, then, you
%       can use max(A, [], 2) to obtain the max for each row.
%









% =========================================================================


end
