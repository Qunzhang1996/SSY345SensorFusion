function[x, P] = mu_g(x, P, yacc, Ra, g0)
    hx=Qq(x)'*g0;
    [Q0, Q1, Q2, Q3]=dQqdq(x);
    Hx=[Q0, Q1, Q2, Q3]*g0;
    S=Hx*P*Hx'+Ra;
    KK=P*Hx'*inv(S);
    x=x+KK*(yacc-hx);
    P=P-KK*S*KK';
end