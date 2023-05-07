function [SP,W] = sigmaPoints(x, P, type)
% SIGMAPOINTS computes sigma points, either using unscented transform or
% using cubature.
%
%Input:
%   x           [n x 1] Prior mean
%   P           [n x n] Prior covariance
%
%Output:
%   SP          [n x 2n+1] UKF, [n x 2n] CKF. Matrix with sigma points
%   W           [1 x 2n+1] UKF, [1 x 2n] UKF. Vector with sigma point weights 
%
n=length(x);
sqrtP=sqrtm(P);
    switch type        
        case 'UKF'
        SP=zeros(n,2*n+1);
        W=zeros(1,2*n+1);
        W(:,1)=1-n/3;
        SP(:,1)=x;
        for i=1:n
            W(:,i+1)=(1-W(:,1))/(2*n);
            W(:,n+i+1)=(1-W(:,1))/(2*n);
            SP(:,i+1)=x+sqrt(n/(1-W(:,1)))*sqrtP(:,i);
            SP(:,i+n+1)=x-sqrt(n/(1-W(:,1)))*sqrtP(:,i);
        end
            % your code     
        case 'CKF'
        SP=zeros(n,2*n);
        W=zeros(1,2*n);  
            % your code
        for i=1:n
            W(:,i)=1/(2*n);
            W(:,i+n)=1/(2*n);
            SP(:, i) = x+sqrt(n)*sqrtP(:,i);
            SP(:, i+n) = x-sqrt(n)*sqrtP(:,i);
        end
        otherwise
            error('Incorrect type of sigma point')
    end

end