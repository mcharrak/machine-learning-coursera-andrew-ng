function [error_train, error_val] = ...
    learningCurve(X, y, Xval, yval, lambda)
%LEARNINGCURVE Generates the train and cross validation set errors needed 
%to plot a learning curve
%   [error_train, error_val] = ...
%       LEARNINGCURVE(X, y, Xval, yval, lambda) returns the train and
%       cross validation set errors for a learning curve. In particular, 
%       it returns two vectors of the same length - error_train and 
%       error_val. Then, error_train(i) contains the training error for
%       i examples (and similarly for error_val(i)).
%
%   In this function, you will compute the train and test errors for
%   dataset sizes from 1 up to m. In practice, when working with larger
%   datasets, you might want to do this in larger intervals.
%

% Number of training examples
m = size(X, 1);

% Number of loops for denoising error terms
n_loops = 20;

% You need to return these values correctly
error_train_matrix = zeros(m, n_loops);
error_val_matrix   = zeros(m, n_loops);

% ====================== YOUR CODE HERE ======================
% Instructions: Fill in this function to return training errors in 
%               error_train and the cross validation errors in error_val. 
%               i.e., error_train(i) and 
%               error_val(i) should give you the errors
%               obtained after training on i examples.
%
% Note: You should evaluate the training error on the first i training
%       examples (i.e., X(1:i, :) and y(1:i)).
%
%       For the cross-validation error, you should instead evaluate on
%       the _entire_ cross validation set (Xval and yval).
%
% Note: If you are using your cost function (linearRegCostFunction)
%       to compute the training and cross validation error, you should 
%       call the function with the lambda argument set to 0. 
%       Do note that you will still need to use lambda when running
%       the training to obtain the theta parameters.
%
% Hint: You can loop over the examples with the following:
%
for i = 1:m
    for j = 1:n_loops 

        % fetch i random examples for each set (training and validation)
        indices = randperm(m);
        train_indices_subset = indices(1:i);
        X_random_subset = X(train_indices_subset,:);
        y_random_subset = y(train_indices_subset);
        
        indices = randperm(m);
        val_indices_subset = indices(1:i);
        Xval_random_subset = Xval(val_indices_subset,:);
        yval_random_subset = yval(val_indices_subset);
     
        % fit and fetch theta
        [theta] = trainLinearReg(X_random_subset, y_random_subset, lambda);

        % calculate error terms for train-set and CV-set
        % for evaluation we calculate the error for a fixed model assumption
        lambda_eval = 0;
        
        [error_train_matrix(i,j), ~] = linearRegCostFunction(X_random_subset, ...
        y_random_subset, theta, lambda_eval);
    
        [error_val_matrix(i,j), ~] = linearRegCostFunction(Xval_random_subset, ...
        yval_random_subset, theta, lambda_eval);
    
    end
end    

% mean(A,2) means that once we have taken the mean operation, the 2nd
% dimension (here the columns will be collapsed to a column vector
error_train = mean(error_train_matrix,2);
error_val = mean(error_val_matrix,2);

% -------------------------------------------------------------

% =========================================================================

end
