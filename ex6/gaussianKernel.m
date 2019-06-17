function sim = gaussianKernel(x1, x2, sigma)
%RBFKERNEL returns a radial basis function kernel between x1 and x2
%   sim = gaussianKernel(x1, x2) returns a gaussian kernel between x1 and x2
%   and returns the value in sim

% Ensure that x1 and x2 are column vectors
x1 = x1(:); x2 = x2(:);

% You need to return the following variables correctly.


% ====================== YOUR CODE HERE ======================
% Instructions: Fill in this function to return the similarity between x1
%               and x2 computed using a Gaussian kernel with bandwidth
%               sigma
%
%
% difference between x1 and x2 
diff = x1-x2;
% dot-product is equal to squared-norm because norm^2 = sqrt(dot)^2
dot = diff'*diff;

var = sigma^2;

%fraction
frac = (dot/(2*var));

sim = exp(-frac);

% =============================================================
    
end
