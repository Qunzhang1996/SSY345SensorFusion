function [x,Y]=normPlot(mu,variance)
%GENLINEARSTATESEQUENCE generates an N-long sequence of states using a 
%    Gaussian prior and a linear Gaussian process model
%
%Input:
%   mu         mean
%   X_true     true value of X
%   X_estimate          estimate value from kalman filter
%   k           sample time
%   value       domain 
%
%Output:
%   Y           normpdf of Y
%
% variance=(X_true(:,k+1)-X_estimate(:,k))^2;
x=[mu-3*sqrt(variance):sqrt(variance)/10:mu+3*sqrt(variance)];
Y=normpdf(x,mu,sqrt(variance));
end