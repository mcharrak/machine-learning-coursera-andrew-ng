function plotDecisionBoundary(theta, X, y)
%PLOTDECISIONBOUNDARY Plots the data points X and y into a new figure with
%the decision boundary defined by theta
%   PLOTDECISIONBOUNDARY(theta, X,y) plots the data points with + for the 
%   positive examples and o for the negative examples. X is assumed to be 
%   a either 
%   1) Mx3 matrix, where the first column is an all-ones column for the 
%      intercept.
%   2) MxN, N>3 matrix, where the first column is all-ones

% Plot Data
plotData(X(:,2:3), y);
hold on

% checks if X has <= 3 columns as this is for the case when we use a linear function 
% as sigmoid input. -> w/o newly created features
if size(X, 2) <= 3
    % Only need 2 points to define a line, so choose two endpoints
    plot_x = [min(X(:,2))-2,  max(X(:,2))+2];

    % Calculate the decision boundary line
    plot_y = (-1./theta(3)).*(theta(2).*plot_x + theta(1));
    
    %plot_y = (-1/theta(3)) * (theta(1) + theta(2) * plot_x );

    % Plot, and adjust axes for better viewing
    plot(plot_x, plot_y)
    
    % Legend, specific for the exercise
    legend('Admitted', 'Not admitted', 'Decision Boundary')
    axis([30, 100, 30, 100])
else
    % Here is the grid range
    % Creates a vector where max value is -1.0 and max value is 1.5 and we
    % have a total of 100 values in that vector
    u = linspace(-1, 1.5, 50);
    v = linspace(-1, 1.5, 50);

    z = zeros(length(u), length(v));
    % Evaluate z = theta*x over the grid
    for i = 1:length(u)
        for j = 1:length(v)
            % we use new pairs of u=x1_feature and v=x2_feature, then we
            % make use of the function mapFeature i.o. to create additional
            % new features {1,x1,x1^2,x2^2,x1*x2,...,x1^6*x2^,etc.} and then 
            % we can caluclate the dot-product with theta (is 28x1 vector)
            
            % we just plot the argument of the sigmoid function because
            % this describes the decision boundary
            z(i,j) = mapFeature(u(i), v(j))*theta;
        end
    end
    z = z'; % important to transpose z before calling contour

    % Plot z = 0
    % Notice you need to specify the range [0, 0] e.g.
    % To draw contour lines at a single height k, we have to specify levels 
    % as a two-element row vector [k k] in our case we are interested 
    % about the decision boundary which is located at z=0 :)
    contour(u, v, z, [0, 0], 'LineWidth', 2)
end
hold off

end
