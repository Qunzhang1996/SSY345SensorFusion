function [hx, Hx] = dualBearingMeasurement(x, s1, s2)
%DUOBEARINGMEASUREMENT calculates the bearings from two sensors, located in 
%s1 and s2, to the position given by the state vector x. Also returns the
%Jacobian of the model at x.
%
%Input:
%   x           [n x 1] State vector, the two first element are 2D position
%   s1          [2 x 1] Sensor position (2D) for sensor 1
%   s2          [2 x 1] Sensor position (2D) for sensor 2
%
%Output:
%   hx          [2 x 1] measurement vector
%   Hx          [2 x n] measurement model Jacobian
%
% NOTE: the measurement model assumes that in the state vector x, the first
% two states are X-position and Y-position.

% Your code here
n=size(x,1);
y1=x(2,:)-s1(2);
y2=x(2,:)-s2(2);
x1=x(1,:)-s1(1);
x2=x(1,:)-s2(1);
hx=[atan2(y1,x1);
    atan2(y2,x2)];
    
Hx =[-y1/(x1^2+y1^2),x1/(x1^2+y1^2),zeros(1,n-2);
    -y2/(x2^2+y2^2),x2/(x2^2+y2^2),zeros(1,n-2)];
end