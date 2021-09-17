function [J, grad] = costFunctionReg(theta, X, y, lambda)
%COSTFUNCTIONREG Compute cost and gradient for logistic regression with regularization
%   J = COSTFUNCTIONREG(theta, X, y, lambda) computes the cost of using
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

z=theta'*X';
h=(1./(1+exp(-z)));
j=0;

for i=1:m
    J=(J+(-y(i)*log(h(i))-(1-y(i))*log(1-h(i)))); 
end
for k=1:size(theta)
   j=(j+theta(k)^2);
end
j=lambda*0.5*j;
J=J+j;

for k=1:size(theta)
    for i=1:m
        
        grad(k)=(grad(k)+(h(i)-y(i))*X(i,k));
        if k==1
           p=0;
        else 
        p=lambda*theta(k);
        end
    end
    grad(k)=grad(k)+p;
end

J=(1/m)*J
grad=grad*(1/m)





% =============================================================

end
