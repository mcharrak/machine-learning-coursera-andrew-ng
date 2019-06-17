function [J, grad] = cofiCostFunc(params, Y, R, num_users, num_movies, ...
                                  num_features, lambda)
%COFICOSTFUNC Collaborative filtering cost function
%   [J, grad] = COFICOSTFUNC(params, Y, R, num_users, num_movies, ...
%   num_features, lambda) returns the cost and gradient for the
%   collaborative filtering problem.
%

% % Unfold the X and Theta matrices from params
% Reshapes the squeezed params-vector into original shape (as matrix - 2D)
X = reshape(params(1:num_movies*num_features), num_movies, num_features);
Theta = reshape(params(num_movies*num_features+1:end), ...
                num_users, num_features);


            
% You need to return the following values correctly
J = 0;
X_grad = zeros(size(X));
Theta_grad = zeros(size(Theta));

% ====================== YOUR CODE HERE ======================
% Instructions: Compute the cost function and gradient for collaborative
%               filtering. Concretely, you should first implement the cost
%               function (without regularization) and make sure it is
%               matches our costs. After that, you should implement the 
%               gradient and use the checkCostFunction routine to check
%               that the gradient is correct. Finally, you should implement
%               regularization.
%
% Notes: X - num_movies  x num_features matrix of movie features
%        Theta - num_users  x num_features matrix of user features
%        Y - num_movies x num_users matrix of user ratings of movies
%        R - num_movies x num_users matrix, where R(i, j) = 1 if the 
%            i-th movie was rated by the j-th user
%
% Tip: idx = find(R(i,:)==1) is a list of all the users that rated movie i
%
% You should set the following variables correctly:
%
%        X_grad - num_movies x num_features matrix, containing the 
%                 partial derivatives w.r.t. to each element of X
%        Theta_grad - num_users x num_features matrix, containing the 
%                     partial derivatives w.r.t. to each element of Theta
%


for j = 1:num_users
    % get the appropriate movies for user j    
    % fetch voting column of user j
    voting_j = R(:,j);
    % find movies i user j has rated
    idx_j = find(voting_j == 1);
    % fetch all features for movies of user j 
    X_j = X(idx_j,:);
    Y_j = Y(idx_j,j);
    % select correct theta row for user j
    Theta_j = Theta(j,:);
    
    % get hypothesis
    hypo_j = X_j*Theta_j';
    % calc difference
    diff_j = hypo_j - Y_j;
    % square differences
    diff_sq_j = diff_j.^2;
    cost_j = sum(diff_sq_j);
    J = J + cost_j;
    
    % calculate Theta_j gradient
    Theta_grad_j = diff_j' * X_j + lambda * Theta_j;     
    
    % save result of Theta_grad_j to Theta_grad matrix
    Theta_grad(j,:) = Theta_grad_j;
end

% calculate regularization terms
reg_Theta = sum(Theta(:).^2);
reg_X = sum(X(:).^2);

reg_total = 0.5*lambda*(reg_Theta+reg_X);

J = 0.5*J + reg_total;

for i = 1:num_movies

    % fetch binary voting vector for movie i
    % voting_i is :(1xn)
    voting_i = R(i,:);
    
    % find index of users that rated movie i
    % idx_i is : (1xa) where a is # of users who have rated movie i
    idx_i = find(voting_i == 1);

    % fetch correct users paramaters Theta using idx_i filter
    
    % Theta_i is (axn)
    Theta_i = Theta(idx_i,:);
    
    % Y_i is (1xa)
    Y_i = Y(i,idx_i);
    
    % select correct feature row for movie i
    X_i = X(i,:);
    
    % get hypothesis
    hypo_i = X_i * Theta_i';
    % calc difference
    diff_i = hypo_i - Y_i;

    X_grad_i = diff_i * Theta_i + lambda * X_i;
    
    % save result of X_grad_i to X_grad matrix
    X_grad(i,:) = X_grad_i;    
end


% =============================================================

grad = [X_grad(:); Theta_grad(:)];

end
