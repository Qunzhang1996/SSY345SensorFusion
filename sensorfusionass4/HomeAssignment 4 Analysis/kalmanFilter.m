function [X, P, X_pred, P_pred,VK] = kalmanFilter(Y, x_0, P_0, A, Q, H, R)
%KALMANFILTER Filters measurements sequence Y using a Kalman filter. 
%
%Input:
%   Y           [m x N] Measurement sequence
%   x_0         [n x 1] Prior mean
%   P_0         [n x n] Prior covariance
%   A           [n x n] State transition matrix
%   Q           [n x n] Process noise covariance
%   H           [m x n] Measurement model matrix
%   R           [m x m] Measurement noise covariance
%
%Output:
%   x           [n x N] Estimated state vector sequence
%   P           [n x n x N] Filter error convariance
%

%% Parameters


n = length(x_0);
%%%%%%%%%%%%to detect NAN%%%%%%%%%%%%%%
k=length(Y);
newData=[];
for i=1:k
    if ~isnan(Y(:,i))%remove NAN data
        newData=[newData,Y(:,i)];
    end
end
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

Y=newData;
N = size(Y,2);
m = size(Y,1);

%% Data allocation
X_pred = zeros(n,N);
P_pred = zeros(n,n,N);
X = zeros(n,N);
P = zeros(n,n,N);
VK=zeros(m,N);
for i=1:N
    [x_predict, P_predict] = linearPrediction(x_0, P_0, A, Q);
    X_pred(:,i)=x_predict;
    P_pred(:,:,i)=P_predict;
    [x_0, P_0,vk] = linearUpdate(x_predict, P_predict, Y(:,i), H, R);
    X(:,i)=x_0;
    VK(:,i)=vk;
    P(:,:,i)=P_0;
end
end
