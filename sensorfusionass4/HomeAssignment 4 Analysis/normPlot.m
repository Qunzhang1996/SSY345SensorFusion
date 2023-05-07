function [x,Y]=normPlot(value,mu,variance)
%GENLINEARSTATESEQUENCE generates an N-long sequence of states using a 
%    Gaussian prior and a linear Gaussian process model
%
%Input:
%   mu         mean
%   X_true     true value of X
%   X_estimate          estimate value from kalman filter
%   variance=(X_true(:,k+1)-X_estimate(:,k))^2;
%   k           sample time
%   value       domain 
%
%Output:
%   Y           normpdf of Y
value=3*sqrt(variance);
x=[mu-value:0.1:mu+value];
Y=normpdf(x,mu,sqrt(variance));
end