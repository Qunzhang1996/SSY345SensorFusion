function [x, P] = mu_m(x, P, mag, m0,Rm)
    hx=Qq(x)'*m0;
    [Q0, Q1, Q2, Q3]=dQqdq(x);
    Hx=[Q0'*m0 Q1'*m0 Q2'*m0 Q3'*m0];
    S=Hx*P*Hx'+Rm;
    KK=P*Hx'*inv(S);
    x=x+KK*(mag-hx);
    P=P-KK*S*KK';
end