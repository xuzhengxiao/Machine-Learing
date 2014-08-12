function [J, grad] = costFunctionReg(theta, X, y, lambda)
%COSTFUNCTIONREG Compute cost and gradient for logistic regression with regularization
%   J = COSTFUNCTIONREG(theta, X, y, lambda) computes the cost of using
%   theta as the parameter for regularized logistic regression and the
%   gradient of the cost w.r.t. to the parameters. 

% Initialize some useful values
m = length(y); % number of training examples ->118

% You need to return the following variables correctly 
J = 0;
grad = zeros(size(theta)); %size(grad)=n*1;->28

% ====================== YOUR CODE HERE ======================
% Instructions: Compute the cost of a particular choice of theta.
%               You should set J to the cost.
%               Compute the partial derivatives and set grad to the partial
%               derivatives of the cost w.r.t. each parameter in theta
n=size(theta);
H = sigmoid(X*theta);%size(H)=m*1
J = 1/m*sum((-y).*log(H)-(1-y).*log(1-H))+lambda/(2*m)*(theta(2:n))'*theta(2:n);
grad(1)=1/m*(sum(H-y));
grad(2:n) = 1/m*((X(:,2:n))'*(H-y))+lambda/m*theta(2:n);







% =============================================================

end
