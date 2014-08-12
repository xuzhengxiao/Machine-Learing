function [J, grad] = linearRegCostFunction(X, y, theta, lambda)
%LINEARREGCOSTFUNCTION Compute cost and gradient for regularized linear 
%regression with multiple variables
%   [J, grad] = LINEARREGCOSTFUNCTION(X, y, theta, lambda) computes the 
%   cost of using theta as the parameter for linear regression to fit the 
%   data points in X and y. Returns the cost in J and the gradient in grad

% Initialize some useful values
m = length(y); % number of training examples m->12

% You need to return the following variables correctly 
J = 0;
grad = zeros(size(theta));%grad->2*1

% ====================== YOUR CODE HERE ======================
% Instructions: Compute the cost and gradient of regularized linear 
%               regression for a particular choice of theta.
%
%               You should set J to the cost and grad to the gradient.
%X->12*2 y->12*1 theta->2*1 lambda->1*1
H=X*theta;
J=1/(2*m)*sum((H-y).^2)+lambda/(2*m)*sum(theta(2:end).^2);%J->1*1
grad(1)= 1/m*sum(H-y);
grad(2:end)=1/m*((X(:,2:end))'*(H-y))+lambda/m*theta(2:end);










% =========================================================================

grad = grad(:);

end
