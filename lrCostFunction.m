function [J, grad] = lrCostFunction(theta, X, y, lambda)
%LRCOSTFUNCTION Compute cost and gradient for logistic regression with 
%regularization
%   J = LRCOSTFUNCTION(theta, X, y, lambda) computes the cost of using
%   theta as the parameter for regularized logistic regression and the
%   gradient of the cost w.r.t. to the parameters. 

% Initialize some useful values
m = length(y); % number of training examples

% You need to return the following variables correctly 
J = 0;
grad = zeros(size(theta));

% ====================== YOUR CODE HERE ======================
% Instructions: Compute the cost of a particular choice of theta.
%               You should set J to the cost.
%               Compute the partial derivatives and set grad to the partial
%               derivatives of the cost w.r.t. each parameter in theta
%
% Hint: The computation of the cost function and gradients can be
%       efficiently vectorized. For example, consider the computation
%
%           sigmoid(X * theta)
%
%       Each row of the resulting matrix will contain the value of the
%       prediction for that example. You can make use of this to vectorize
%       the cost function and gradient computations. 
%
% Hint: When computing the gradient of the regularized cost function, 
%       there're many possible vectorized solutions, but one solution
%       looks like:
%           grad = (unregularized gradient for logistic regression)
%           temp = theta; 
%           temp(1) = 0;   % because we don't add anything for j = 0  
%           grad = grad + YOUR_CODE_HERE (using the temp variable)
%
thetaTransX = X*theta;

OneMatrix = ones(size(thetaTransX));

ifYEquals1=log(sigmoid(thetaTransX'));
ifYEquals0=OneMatrix' - (sigmoid(thetaTransX'));
coeffy=-1*y;

yMin1 =  y-1;

J =(1/m)*(coeffy'*ifYEquals1'+yMin1'*log(ifYEquals0'));

theta(1) = 0;

J = J + lambda/(2*m)*(sum(theta.^2));

grad = (1/m)*(sigmoid(thetaTransX')-y')*X;

grad = grad + lambda/m*theta';

% =============================================================

grad = grad(:);

end
