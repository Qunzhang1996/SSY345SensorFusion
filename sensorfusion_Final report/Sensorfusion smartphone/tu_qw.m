function [x,P] =tu_qw(x, P, omega, T, Rw)
F=eye(size(x,1))+T/2*Somega(omega);
G=T/2*Sq(x);
x=F*x;
P=F*P*F'+G*Rw*G';
end