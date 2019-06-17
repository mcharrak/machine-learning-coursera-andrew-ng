function visualizeBoundary(X, y, model, varargin)
%VISUALIZEBOUNDARY plots a non-linear decision boundary learned by the SVM
%   VISUALIZEBOUNDARYLINEAR(X, y, model) plots a non-linear decision 
%   boundary learned by the SVM and overlays the data on it

% Plot the training data on top of the boundary
plotData(X, y)

% Make classification predictions over a grid of values
x1plot = linspace(min(X(:,1)), max(X(:,1)), 100)';
x2plot = linspace(min(X(:,2)), max(X(:,2)), 100)';

% X1: each row is a copy of x1plot
% X2: each column is a copy of x2plot

% size(X1): no. cols = length(x1plot), no. rows = length(x2plot)
% size(X1) equals size(X2)
[X1, X2] = meshgrid(x1plot, x2plot);
vals = zeros(size(X1));

% we loop over ea. col. of X1 because ea. col. contains all same values
for i = 1:size(X1, 2)
   % now we choose a single column because it contains the vector x2plot 
   this_X = [X1(:, i), X2(:, i)];
   vals(:, i) = svmPredict(model, this_X);
end

% Plot the SVM boundary
hold on
contour(X1, X2, vals, [0.5 0.5], 'b');
hold off;

end
