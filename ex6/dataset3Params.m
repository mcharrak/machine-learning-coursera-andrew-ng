function [C, sigma] = dataset3Params(X, y, Xval, yval)
%DATASET3PARAMS returns your choice of C and sigma for Part 3 of the exercise
%where you select the optimal (C, sigma) learning parameters to use for SVM
%with RBF kernel
%   [C, sigma] = DATASET3PARAMS(X, y, Xval, yval) returns your choice of C and 
%   sigma. You should complete this function to return the optimal C and 
%   sigma based on a cross-validation set.
%

% You need to return the following variables correctly.
C = 1;
sigma = 0.3;

% ====================== YOUR CODE HERE ======================
% Instructions: Fill in this function to return the optimal C and sigma
%               learning parameters found using the cross validation set.
%               You can use svmPredict to predict the labels on the cross
%               validation set. For example, 
%                   predictions = svmPredict(model, Xval);
%               will return the predictions on the cross validation set.
%
%  Note: You can compute the prediction error using 
%        mean(double(predictions ~= yval))
%

C_vec =  [0.01, 0.03, 0.1, 0.3, 1, 3, 10, 30];
sigma_vec = [0.01, 0.03, 0.1, 0.3, 1, 3, 10, 30];
n_runs = length(C_vec) * length(sigma_vec);

% create global index and results matrix for saving results
glob_idx = 1;
% results: saves experimental CV error
% col1: value of C // col2: value of sigma // col3: validation error
results = zeros(n_runs,3);

for i = 1:numel(C_vec)
    C = C_vec(i);
    for j = 1:numel(sigma_vec)
        sigma = sigma_vec(j);

        % Train the SVM
        model = svmTrain(X, y, C, @(x1, x2)gaussianKernel(x1, x2, sigma));

        % Fetch predictions for Xval
        predictions = svmPredict(model, Xval);
        
        % Compute misclassification error rate (LOWER IS BETTER) on CV-set
        val_err = mean(double(predictions ~= yval));
        
        % Save results in results_matrix each row is: [C,sigma,val_error]
        results(glob_idx,:) = [C,sigma,val_err];
        
        % Increase global index
        glob_idx = glob_idx + 1;
        
    end
end

% choose row where 3rd column value (validation error) is the smallest
% find minimum validation error and the corresponding row-index in results
[~, min_index] = min(results(:,3));

opt_vals = results(min_index, :); % contains [C sigma val_error]

% fetch all 3 results
C = opt_vals(1);
sigma = opt_vals(2);
val_error = opt_vals(3);

% =========================================================================

end
