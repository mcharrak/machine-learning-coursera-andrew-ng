function Z = projectData(X, U, K)
%PROJECTDATA Computes the reduced data representation when projecting only 
%on to the top k eigenvectors
%   Z = projectData(X, U, K) computes the projection of 
%   the normalized inputs X into the reduced dimensional space spanned by
%   the first K columns of U. It returns the projected examples in Z.
%

% You need to return the following variables correctly.
Z = zeros(size(X, 1), K);

% ====================== YOUR CODE HERE ======================
% Instructions: Compute the projection of the data using only the top K 
%               eigenvectors in U (first K columns). 
%               For the i-th example X(i,:), the projection on to the k-th 
%               eigenvector is given as follows:
%                    x = X(i, :)';
%                    projection_k = x' * U(:, k);
%

% fetch the first k principal components (i.e. eigenvectors)
Ureduce = U(:,1:K);

% Ureduce is of dimension nxk where n is dimension of example x (here n=2)
% Ureduce is thus : 2x1
% X is of dimension mxn where m is number of examples
% X is thus : 50x2


% Z must be of dimension mxk (here 50x1)
Z = X * Ureduce;

% =============================================================

end
