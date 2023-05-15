%% %Question1 Smoothing

clc
clear all
%% True track
% Sampling period
T = 0.1;
% Length of time sequence
K = 600;
% Allocate memory
omega = zeros(1,K+1);
% Turn rate
omega(150:450) = -pi/301/T;
% Initial state
x0 = [0 0 20 0 omega(1)]';
% Allocate memory
X = zeros(length(x0),K+1);
X(:,1) = x0;
% Create true track
for i=2:K+1
% Simulate
X(:,i) = coordinatedTurnMotion(X(:,i-1), T);
% Set turn−rate
X(5,i) = omega(i);
end
%%
x_0=[0 0 0 0 0]';N=length(X);
P_0=diag([10^2 10^2 10^2 (5*pi/180)^2 (pi/180)^2]);
s1=[300 -100]';s2=[300 -300]';
sigma_phi1=pi/180;sigma_phi2=pi/180;
R=[sigma_phi1^2 0;0 sigma_phi2^2];
Tau=[0 0 1 0 0;0 0 0 0 1]';
f=@(x,T)coordinatedTurnMotion(x, T);
h=@(x,T) dualBearingMeasurement(x, s1, s2);
Y1 = genNonLinearMeasurementSequence(X, h, R);
%%
sigma_v=0.1;sigma_w=0.1*pi/180;
Q=Tau*[sigma_v^2 0;0 sigma_w^2]*Tau';
Point=[];
for i=1:600
    [x1, y1] = getPosFromMeasurement(Y1(1,i), Y1(2,i), s1, s2);
    Point=[Point;[x1, y1]];
end
% [xf_tunning, Pf, xp, Pp] = nonLinearKalmanFilter(Y1, x_0, P_0, f, Q, h, R, 'CKF');
[xs, Ps, xf_tunning, Pf, xp, Pp] = nonLinRTSsmoother(Y1, x_0, P_0, f, T, Q, h, R, @sigmaPoints, 'CKF');
%%
figure('Position',[300 300 600 400]);hold on;
plot(s1(1),s1(2),'+r','MarkerSize',14,'LineWidth',2);
plot(s2(1),s2(2),'*g','MarkerSize',14,'LineWidth',2);
plot(X(1,1:end), X(2,1:end),'Color','blue',LineWidth=2);
plot(Point(1:end,1), Point(1:end,2),'--k','Color','m',LineWidth=1);
plot(xf_tunning(1,1:end), xf_tunning(2,1:end),'Color','r',LineWidth=2);
for i=1:5:length(xf_tunning)
    covEllipse = sigmaEllipse2D(xf_tunning(1:2,i),Pf(1:2,1:2,i),3,100);
    plot(covEllipse(1,:),covEllipse(2,:),'black','LineWidth',1);
    fill(covEllipse(1,:),covEllipse(2,:), 'cyan','FaceAlpha', 0.5);
end
axis equal
grid on
legend('sensor1', 'sensor2','true states','measurement','CKF','$x \pm 3\sigma$','Interpreter','latex')
%%
figure('Position',[300 300 600 400]);hold on;
plot(s1(1),s1(2),'+r','MarkerSize',14,'LineWidth',2);
plot(s2(1),s2(2),'*g','MarkerSize',14,'LineWidth',2);
plot(X(1,1:end), X(2,1:end),'Color','blue',LineWidth=2);
plot(Point(1:end,1), Point(1:end,2),'--k','Color','m',LineWidth=1);
plot(xs(1,1:end), xs(2,1:end),'Color','r',LineWidth=2);
for i=1:5:length(xs)
    covEllipse = sigmaEllipse2D(xs(1:2,i),Ps(1:2,1:2,i),3,100);
    plot(covEllipse(1,:),covEllipse(2,:),'black','LineWidth',1);
    fill(covEllipse(1,:),covEllipse(2,:), 'cyan','FaceAlpha', 0.5);
end
axis equal
grid on
legend('sensor1', 'sensor2','true states','measurement','smoothing','$x \pm 3\sigma$','Interpreter','latex')
%%
Norm_tunning=vecnorm(X(1:2,2:end)-xf_tunning(1:2,:));
Norm_smoothing=vecnorm(X(1:2,2:end)-xs(1:2,:));
figure('Position',[300 300 600 400]);hold on;
subplot(3,1,1)
plot(0:599,Norm_tunning)
legend("good tunning data")
subplot(3,1,2)
plot(0:599,Norm_smoothing)
legend("smoothing data")
%%
Y2=Y1;
Y2(:,300)=Y2(:,300)+50*mvnrnd(zeros(length(R),1),R)';
Point=[];
for i=1:600
    [x1, y1] = getPosFromMeasurement(Y2(1,i), Y2(2,i), s1, s2);
    Point=[Point;[x1, y1]];
end
% [xf_tunning, Pf, xp, Pp] = nonLinearKalmanFilter(Y1, x_0, P_0, f, Q, h, R, 'CKF');
[xs_add, Ps_add, xf_tunning_add, Pf_add, xp, Pp] = nonLinRTSsmoother(Y2, x_0, P_0, f, T, Q, h, R, @sigmaPoints, 'CKF');
%%
figure('Position',[300 300 600 400]);hold on;
plot(s1(1),s1(2),'+r','MarkerSize',14,'LineWidth',2);
plot(s2(1),s2(2),'*g','MarkerSize',14,'LineWidth',2);
plot(X(1,1:end), X(2,1:end),'Color','blue',LineWidth=2);
plot(Point(1:end,1), Point(1:end,2),'--k','Color','m',LineWidth=1);
plot(xf_tunning_add(1,1:end), xf_tunning_add(2,1:end),'Color','r',LineWidth=2);
for i=1:5:length(xf_tunning_add)
    covEllipse = sigmaEllipse2D(xf_tunning_add(1:2,i),Pf_add(1:2,1:2,i),3,100);
    plot(covEllipse(1,:),covEllipse(2,:),'black','LineWidth',1);
    fill(covEllipse(1,:),covEllipse(2,:), 'cyan','FaceAlpha', 0.5);
end
axis equal
grid on
legend('sensor1', 'sensor2','true states','measurement','CKF','$x \pm 3\sigma$','Interpreter','latex')
%%
figure('Position',[300 300 600 400]);hold on;
plot(s1(1),s1(2),'+r','MarkerSize',14,'LineWidth',2);
plot(s2(1),s2(2),'*g','MarkerSize',14,'LineWidth',2);
plot(X(1,1:end), X(2,1:end),'Color','blue',LineWidth=2);
plot(Point(1:end,1), Point(1:end,2),'--k','Color','m',LineWidth=1);
plot(xs_add(1,1:end), xs_add(2,1:end),'Color','r',LineWidth=2);
for i=1:5:length(xs)
    covEllipse = sigmaEllipse2D(xs_add(1:2,i),Ps_add(1:2,1:2,i),3,100);
    plot(covEllipse(1,:),covEllipse(2,:),'black','LineWidth',1);
    fill(covEllipse(1,:),covEllipse(2,:), 'cyan','FaceAlpha', 0.5);
end
axis equal
grid on
legend('sensor1', 'sensor2','true states','measurement','smoothing','$x \pm 3\sigma$','Interpreter','latex')
%%
Norm_tunning=vecnorm(X(1:2,2:end)-xf_tunning_add(1:2,:));
Norm_smoothing=vecnorm(X(1:2,2:end)-xs_add(1:2,:));
figure('Position',[300 300 600 400]);hold on;
subplot(3,1,1)
plot(0:599,Norm_tunning)
legend("good tunning data")
subplot(3,1,2)
plot(0:599,Norm_smoothing)
legend("smoothing data")
%% 
% %Question2 Particle filters for linear/Gaussian systems
%% plot traj

Q=1.5;R=3;x_0=2;P_0=8;T=0.1;A=1;H=1;N=31;plotFunc=[];bResample=0;num1=10000;num2=10000;
f = @(x) A*x;
h=@(x) H*x;
X = genLinearStateSequence(x_0, P_0, A, Q, N);
Y = genLinearMeasurementSequence(X, H, R);0.75
%%
num1=4000;num2=4000;
[X_kf, P_kf, X_pred, P_pred,VK] = kalmanFilter(Y, x_0, P_0, A, Q, H, R);
[xfp, Pfp, Xp, Wp] = pfFilter(x_0, P_0, Y, f, Q, h, R, num1, bResample, [], []);
[xfp1, Pfp1, Xp1, Wp1] = pfFilter(x_0, P_0, Y, f, Q, h, R, num2, ~bResample, [], []);
figure('Position',[300 300 600 400]);hold on;
plot(0:N,X,'Color','m',LineWidth=2)
plot(1:N,Y,'--','Color','green',LineWidth=1)
plot(1:N,X_kf,'Color','r',LineWidth=2)
plot(1:N,xfp,'Color','cyan',LineWidth=2)
plot(1:N,xfp1,'--','Color','blue',LineWidth=2)
axis equal
grid on
legend('TRUE STATE','measuerment','KF', '$PF_{without}$','$PF_{resampling}$','Interpreter','latex')
MSE_KF=immse(X_kf,X(:,2:end))
MSE_fp=immse(xfp,X(:,2:end))
MSE_fp1=immse(xfp1,X(:,2:end))
%100 is enough for resampling, however even 100000 is not enough for
%without resampling to catch up with KF
%%
X_PLOT=[1:1:N];
figure('Position',[300 300 600 400]);hold on;
subplot(3,1,1);hold on
plot(1:N,Y,'--','Color','green',LineWidth=1);
plot(0:N,X,'Color','m',LineWidth=2);
errorbar(X_PLOT,X_kf(1:end),X_kf-X(2:end),LineWidth=2);
legend('measuerment','TRUE STATE','KF errorbar','Interpreter','latex')
subplot(3,1,2);hold on
plot(1:N,Y,'--','Color','green',LineWidth=1);
plot(0:N,X,'Color','m',LineWidth=2);
errorbar(X_PLOT,xfp(1:end),xfp-X(2:end),LineWidth=2)
legend('measuerment','TRUE STATE','PF without re errorbar','Interpreter','latex')
subplot(3,1,3);hold on
plot(1:N,Y,'--','Color','green',LineWidth=1);
plot(0:N,X,'Color','m',LineWidth=2);
errorbar(X_PLOT,xfp1(1:end),xfp1-X(2:end),LineWidth=2)
legend('measuerment','TRUE STATE','PF with re errorbar','Interpreter','latex')
% plot(1:N,xfp1,'--','Color','blue',LineWidth=2)
%%
sigma=1;sigma1=1;k1=1;k2=15;k3=30;
figure('Position',[300 300 600 400]);
subplot(2,3,1)
plotPostPdf(k1, Xp(:,:,k1), Wp(:,k1)', X_kf, P_kf,bResample, 1, 0)
subplot(2,3,4)
plotPostPdf(k1, Xp1(:,:,k1), Wp1(:,k1)', X_kf, P_kf, ~bResample, sigma1, 0)
subplot(2,3,2)
plotPostPdf(k2,Xp(:,:,k2), Wp(:,k2)', X_kf, P_kf, bResample, 1, [])
subplot(2,3,5)
plotPostPdf(k2, Xp1(:,:,k2), Wp1(:,k2)', X_kf, P_kf, ~bResample, sigma1, [])
% figure('Position',[300 300 600 400]);
subplot(2,3,3)
plotPostPdf(k3, Xp(:,:,k3), Wp(:,k3)', X_kf, P_kf, bResample, 1, [])
subplot(2,3,6)
plotPostPdf(k3, Xp1(:,:,k3), Wp1(:,k3)', X_kf, P_kf, ~bResample, sigma1, [])
%%
x_02=-20;P_02=2;
[X_kf, P_kf, X_pred, P_pred,VK] = kalmanFilter(Y, x_02, P_02, A, Q, H, R);
[xfp, Pfp, Xp, Wp] = pfFilter(x_02, P_02, Y, f, Q, h, R, num1, bResample, [], []);
[xfp1, Pfp1, Xp1, Wp1] = pfFilter(x_02, P_02, Y, f, Q, h, R, num2, ~bResample, [], []);
figure('Position',[300 300 600 400]);hold on;
plot(0:N,X,'Color','m',LineWidth=2)
plot(1:N,Y,'--','Color','green',LineWidth=2)
plot(1:N,X_kf,'Color','r',LineWidth=2)
plot(1:N,xfp,'Color','cyan',LineWidth=2)
plot(1:N,xfp1,'--','Color','blue',LineWidth=2)
xlim([0 35])
ylim([-20 35])
grid on
legend('TRUE STATE','measurement','KF', '$PF_{without}$','$PF_{resampling}$','Interpreter','latex')
%%
plotfunc=@(i, Xk, Xkmin1, Wk, j)plotPartTrajs(i, Xk, Xkmin1, Wk, j);
num1=100;
figure('Position',[300 300 600 400]);hold on;
[xfp, Pfp, Xp, Wp] = pfFilter(x_0, P_0, Y, f, Q, h, R, num1, bResample, plotfunc, []);
plot(0:N,X,'Color','m',LineWidth=2);
plot(1:N,xfp,'Color','cyan',LineWidth=2);
grid on
ylim([-20 20])
%%
figure('Position',[300 300 600 400]);hold on;
[xfp, Pfp, Xp, Wp] = pfFilter(x_0, P_0, Y, f, Q, h, R, num1, ~bResample, plotfunc, []);
plot(0:N,X,'Color','m',LineWidth=2);
plot(1:N,xfp,'Color','red',LineWidth=2);
grid on
ylim([-20 20])
%% 3.Bicycle tracking in a village
%% question b

clc
clear all
load('Xk.mat');
X=Xk(1,:);
Y=Xk(2,:);
sigma_r=pi/180;R=eye(2)*sigma_r^2;X_0=mvnrnd([2,8],1*eye(2))'
X_kmins1=[X_0 Xk];X_kmins1(:,end)=[];
X_V=Xk-X_kmins1;H=eye(2);
V_measure=genLinearMeasurementSequence(X_V, H, R);


%% question c
%% 
% *P(xk | Y, M)=P(M | xk, Y )P(xk | Y) / P(M|Y) →P(M | xk )*P(xk | Y)* 
%% Question d

T=1;num1=10000;N=length(Xk);
sigma_r=0.05;R=eye(2)*sigma_r^2;X_0=Xk(:,1);
X_kmins1=[X_0 Xk];X_kmins1(:,end)=[];
X_V=Xk-X_kmins1;H=1;

%%
f=@(x)coordinatedTurnMotion(x, T);
h = @(x) H*[x(3,:).*cos(x(4,:));
          x(3,:).*sin(x(4,:))];
sigma_v=0.1;sigma_w=10*pi/180;
Tau=[0 0 1 0 0;0 0 0 0 1]';
Q=Tau*[sigma_v^2 0;0 sigma_w^2]*Tau';
V=vecnorm(X_V);
V_measure=genLinearMeasurementSequence(X_V, H, R);
x_0=[X(1) Y(1) V(1) 0 0]';
P_0=diag([0 0 0 10 10]);
isOnRoad=@(x,y)isOnRoad(x,y);
figure('Position',[300 300 600 400]);hold on;
plot(1:96,V_measure(1,:));
plot(1:96,V_measure(2,:));
grid on
legend('v1','v2')

%%
Q1=diag([0 0 1 1]);x_01=[X(1) Y(1) X(2)-X(1) Y(2)-Y(1)]';P_01=diag([0 0 0 0]);
f1=@(x)constantVelocity(x, T);h1=@(x)cvMeasurement1(x);
[xfp_cv, Pfp_cv, Xp_cv, Wp_cv] = pfFilter(x_01, P_01, V_measure, f1, Q1, h1, R, num1, true, [],isOnRoad);
[xfp1_cv, Pfp1_cv, Xp1_cv, Wp1_cv] = pfFilter(x_01, P_01, V_measure, f1, Q1, h1, R, num1, true, [],[]);
%%
figure('Position',[300 300 600 400]);hold on;
plotbycycle(X, Y, xfp_cv, Pfp_cv, Xp_cv, []);   
%%
figure('Position',[300 300 600 400]);hold on;
plotbycycle(X, Y, xfp1_cv, Pfp1_cv, Xp1_cv, []);   
%%
X_true = genNonLinearStateSequence(x_0, P_0, f, Q, N);
[xfp, Pfp, Xp, Wp] = pfFilter(x_0, P_0, V_measure, f, Q, h, R, num1, true, [],isOnRoad);
[xfp1, Pfp1, Xp1, Wp1] = pfFilter(x_0, P_0, V_measure, f, Q, h, R, num1, true, [],[]);

%%
figure('Position',[300 300 600 400]);hold on;
plotbycycle(X, Y, xfp1, Pfp1,Xp1, []);
%%
figure('Position',[300 300 600 400]);hold on;
plotbycycle(X, Y, xfp, Pfp,Xp, []);
%%
figure('Position',[300 300 600 400]);hold on;
PLT=1;
plotbycycle(X, Y, xfp, Pfp,Xp, PLT);
%% Question e

figure('Position',[300 300 600 400]);hold on;
% axis([0.8 11.2 0.8 9.2])
title('A map of the village','FontSize',20)
% Define the number of random points to generate
num_points = 10000;
% Define the map boundaries
x_min = 1;
x_max = 11;
y_min = 1;
y_max = 9;
% Generate random points within the map boundaries
x_vec = x_min + (x_max - x_min) * rand(num_points, 1);
y_vec = y_min + (y_max - y_min) * rand(num_points, 1);
% Use the isOnRoad function to check which points are on the road
u = isOnRoad(x_vec, y_vec);
% Extract the points that are on the road
onroad_points = [x_vec(u == 1), y_vec(u == 1)];
plot(onroad_points(:, 1), onroad_points(:, 2), '.');
title('Random Points on the Road');
axis equal;
%%
num_check=randi(length(onroad_points))
X_check0=onroad_points(num_check,1);Y_check0=onroad_points(num_check,2);
x_0=[X_check0 Y_check0 V(1) 0 0]';
P_0=diag([0 0 0 10 10]);
[xfp_e, Pfp_e, Xp_e, Wp_e] = pfFilter(x_0, P_0, V_measure, f, Q, h, R, num1, true, [],isOnRoad);
[xfp_e2, Pfp_e2, Xp_e2, Wp_e2] = pfFilter(x_0, P_0, V_measure, f, Q, h, R, num1, true, [],[]);
%%
figure('Position',[300 300 600 400]);hold on;
plotbycycle(X, Y, xfp_e, Pfp_e, Xp_e, []);
%%
figure('Position',[300 300 600 400]);hold on;
plotbycycle(X, Y, xfp_e2, Pfp_e2, Xp_e2, []);
%% 
%% HA4.1.2 Non-linear RTS smoother

function [xs, Ps, xf, Pf, xp, Pp] = nonLinRTSsmoother(Y, x_0, P_0, f, T, Q, h, R, sigmaPoints, type)
%NONLINRTSSMOOTHER Filters measurement sequence Y using a 
% non-linear Kalman filter. 
%
%Input:
%   Y           [m x N] Measurement sequence for times 1,...,N
%   x_0         [n x 1] Prior mean for time 0
%   P_0         [n x n] Prior covariance
%   f                   Motion model function handle
%   T                   Sampling time
%   Q           [n x n] Process noise covariance
%   S           [n x N] Sensor position vector sequence
%   h                   Measurement model function handle
%   R           [n x n] Measurement noise covariance
%   sigmaPoints Handle to function that generates sigma points.
%   type        String that specifies type of non-linear filter/smoother
%
%Output:
%   xf          [n x N]     Filtered estimates for times 1,...,N
%   Pf          [n x n x N] Filter error convariance
%   xp          [n x N]     Predicted estimates for times 1,...,N
%   Pp          [n x n x N] Filter error convariance
%   xs          [n x N]     Smoothed estimates for times 1,...,N
%   Ps          [n x n x N] Smoothing error convariance

% your code here!
% We have offered you functions that do the non-linear Kalman prediction and update steps.
% Call the functions using
% [xPred, PPred] = nonLinKFprediction(x_0, P_0, f, T, Q, sigmaPoints, type);
% [xf, Pf] = nonLinKFupdate(xPred, PPred, Y, S, h, R, sigmaPoints, type);
N=size(Y,2);
m = size(Y,1);
n = length(x_0);
xs=zeros(n,N);Ps=zeros(n,n,N);xf=zeros(n,N);Pf=zeros(n,n,N);xp=zeros(n,N);Pp=zeros(n,n,N);
for k=1:N
    [x_predict, P_predict] = nonLinKFprediction(x_0, P_0, f, T, Q, sigmaPoints, type);
    xp(:,k)=x_predict;
    Pp(:,:,k)=P_predict;
    [x_0, P_0] =nonLinKFupdate(x_predict, P_predict, Y(:,k), h, R, sigmaPoints, type);
    xf(:,k)=x_0;
    Pf(:,:,k)=P_0;
end

Ps_kplus1=Pf(:,:,N);
xs_kplus1=xf(:,N);
xs(:,N)=xf(:,N);
Ps(:,:,N)=Pf(:,:,N);
    for k=N-1:-1:1
        [xs_mid, Ps_mid] = nonLinRTSSupdate(xs_kplus1, Ps_kplus1, xf(:,k), Pf(:,:,k), xp(:,k+1),  Pp(:,:,k+1), f, T, sigmaPoints, type);
        xs_kplus1=xs_mid;
        xs(:,k)=xs_mid;
        Ps_kplus1=Ps_mid;
        Ps(:,:,k)=Ps_mid;
    end
end



function [xs, Ps] = nonLinRTSSupdate(xs_kplus1, Ps_kplus1, xf_k, Pf_k, xp_kplus1, Pp_kplus1, f, T, sigmaPoints, type)
    % Your code here! Copy from previous task!
    % Your code here.
n=size(Pf_k,1);
switch type
        case 'EKF'
           [fx,Fx]=f(xf_k,T);
            % Your EKF code here
            G=Pf_k*Fx'*inv(Pp_kplus1)
            xs=xf_k+G*(xs_kplus1-fx);
            Ps=Pf_k-G*(Pp_kplus1-Ps_kplus1)*G';   
            
        case 'UKF'
            % Your UKF code here
            [SP,W] = sigmaPoints(xf_k, Pf_k, type);
%             [fx,Fx]=f(SP,T);

            P_mid=zeros(n,n);
            for i=1:length(W)
                f_pred(:,i) = f(SP(:,i),T);
                P_mid=P_mid+(SP(:,i)-xf_k)*(f(SP(:,i),T)-xp_kplus1)'*W(i);
            end
            G=P_mid*inv(Pp_kplus1);
            xs=xf_k+G*(xs_kplus1-xp_kplus1);
            Ps=Pf_k-G*(Pp_kplus1-Ps_kplus1)*G';  
            % Make sure the covariance matrix is semi-definite
               
        case 'CKF'
            
            % Your CKF code here
           [SP,W] = sigmaPoints(xf_k, Pf_k, type);
%          [fx,Fx]=f(SP,T);

            P_mid=zeros(n,n);
            for i=1:length(W)
                f_pred(:,i) = f(SP(:,i),T);
                P_mid=P_mid+(SP(:,i)-xf_k)*(f(SP(:,i),T)-xp_kplus1)'*W(i);
            end
            G=P_mid*inv(Pp_kplus1);
            xs=xf_k+G*(xs_kplus1-xp_kplus1);
            Ps=Pf_k-G*(Pp_kplus1-Ps_kplus1)*G'; 
        otherwise
            error('Incorrect type of non-linear Kalman filter')
    end
end

function [x, P] = nonLinKFprediction(x, P, f, T, Q, sigmaPoints, type)
%NONLINKFPREDICTION calculates mean and covariance of predicted state
%   density using a non-linear Gaussian model.
%
%Input:
%   x           [n x 1] Prior mean
%   P           [n x n] Prior covariance
%   f           Motion model function handle
%   T           Sampling time
%   Q           [n x n] Process noise covariance
%   sigmaPoints Handle to function that generates sigma points.
%   type        String that specifies the type of non-linear filter
%
%Output:
%   x           [n x 1] predicted state mean
%   P           [n x n] predicted state covariance
%

    switch type
        case 'EKF'

            % Evaluate motion model
            [fx, Fx] = f(x,T);
            % State prediction
            x = fx;
            % Covariance prediciton
            P = Fx*P*Fx' + Q;
            % Make sure P is symmetric
            P = 0.5*(P + P');

        case 'UKF'

            % Predict
            [x, P] = predictMeanAndCovWithSigmaPoints(x, P, f, T, Q, sigmaPoints, type);

            if min(eig(P))<=0
                [v,e] = eig(P);
                emin = 1e-3;
                e = diag(max(diag(e),emin));
                P = v*e*v';
            end

        case 'CKF'

            % Predict
            [x, P] = predictMeanAndCovWithSigmaPoints(x, P, f, T, Q, sigmaPoints, type);

        otherwise
            error('Incorrect type of non-linear Kalman filter')
    end
end

function [x, P] = nonLinKFupdate(x, P, y, h, R, sigmaPoints, type)
%NONLINKFUPDATE calculates mean and covariance of predicted state
%   density using a non-linear Gaussian model.
%
%Input:
%   x           [n x 1] Prior mean
%   P           [n x n] Prior covariance
%   y           [m x 1] measurement vector
%   s           [2 x 1] sensor position vector
%   h           Measurement model function handle
%   R           [n x n] Measurement noise covariance
%   sigmaPoints Handle to function that generates sigma points.
%   type        String that specifies the type of non-linear filter
%
%Output:
%   x           [n x 1] updated state mean
%   P           [n x n] updated state covariance
%


switch type
    case 'EKF'
        
        % Evaluate measurement model
        [hx, Hx] = h(x);
        
        % Innovation covariance
        S = Hx*P*Hx' + R;
        % Kalman gain
        K = (P*Hx')/S;
        
        % State update
        x = x + K*(y - hx);
        % Covariance update
        P = P - K*S*K';
        
        % Make sure P is symmetric
        P = 0.5*(P + P');
        
    case 'UKF'

        % Update mean and covariance
        [x, P] = updateMeanAndCovWithSigmaPoints(x, P, y, h, R, sigmaPoints, type);
        
        if min(eig(P))<=0
            [v,e] = eig(P);
            emin = 1e-3;
            e = diag(max(diag(e),emin));
            P = v*e*v';
        end
        
    case 'CKF'

        % Update mean and covariance
        [x, P] = updateMeanAndCovWithSigmaPoints(x, P, y, h, R, sigmaPoints, type);
        
    otherwise
        error('Incorrect type of non-linear Kalman filter')
end

end


function [x, P] = predictMeanAndCovWithSigmaPoints(x, P, f, T, Q, sigmaPoints, type)
%
%PREDICTMEANANDCOVWITHSIGMAPOINTS computes the predicted mean and covariance
%
%Input:
%   x           [n x 1] mean vector
%   P           [n x n] covariance matrix 
%   f           measurement model function handle
%   T           sample time
%   Q           [m x m] process noise covariance matrix
%
%Output:
%   x           [n x 1] Updated mean
%   P           [n x n] Updated covariance
%

    % Compute sigma points
    [SP,W] = sigmaPoints(x, P, type);

    % Dimension of state and number of sigma points
    [n, N] = size(SP);

    % Allocate memory
    fSP = zeros(n,N);

    % Predict sigma points
    for i = 1:N
        [fSP(:,i),~] = f(SP(:,i),T);
    end

    % Compute the predicted mean
    x = sum(fSP.*repmat(W,[n, 1]),2);

    % Compute predicted covariance
    P = Q;
    for i = 1:N
        P = P + W(i)*(fSP(:,i)-x)*(fSP(:,i)-x)';
    end

    % Make sure P is symmetric
    P = 0.5*(P + P');

end

function [x, P] = updateMeanAndCovWithSigmaPoints(x, P, y, h, R, sigmaPoints, type)
%
%UPDATEGAUSSIANWITHSIGMAPOINTS computes the updated mean and covariance
%
%Input:
%   x           [n x 1] Prior mean
%   P           [n x n] Prior covariance
%   y           [m x 1] measurement
%   s           [2 x 1] sensor position
%   h           measurement model function handle
%   R           [m x m] measurement noise covariance matrix
%
%Output:
%   x           [n x 1] Updated mean
%   P           [n x n] Updated covariance
%

    % Compute sigma points
    [SP,W] = sigmaPoints(x, P, type);

    % Dimension of measurement
    m = size(R,1);

    % Dimension of state and number of sigma points
    [n, N] = size(SP);

    % Predicted measurement
    yhat = zeros(m,1);
    hSP = zeros(m,N);
    for i = 1:N
        [hSP(:,i),~] = h(SP(:,i));
        yhat = yhat + W(i)*hSP(:,i);
    end

    % Cross covariance and innovation covariance
    Pxy = zeros(n,m);
    S = R;
    for i=1:N
        Pxy = Pxy + W(i)*(SP(:,i)-x)*(hSP(:,i)-yhat)';
        S = S + W(i)*(hSP(:,i)-yhat)*(hSP(:,i)-yhat)';
    end

    % Ensure symmetry
    S = 0.5*(S+S');

    % Updated mean
    x = x+Pxy*(S\(y-yhat));
    P = P - Pxy*(S\(Pxy'));

    % Ensure symmetry
    P = 0.5*(P+P');

end
%% HA4.1.5 Particle filter

function [xfp, Pfp, Xp, Wp] = pfFilter(x_0, P_0, Y, proc_f, proc_Q, meas_h, meas_R, ...
                             N, bResample, plotFunc,isOnRoad)
%PFFILTER Filters measurements Y using the SIS or SIR algorithms and a
% state-space model.
%
% Input:
%   x_0         [n x 1] Prior mean
%   P_0         [n x n] Prior covariance
%   Y           [m x K] Measurement sequence to be filtered
%   proc_f      Handle for process function f(x_k-1)
%   proc_Q      [n x n] process noise covariance
%   meas_h      Handle for measurement model function h(x_k)
%   meas_R      [m x m] measurement noise covariance
%   N           Number of particles
%   bResample   boolean false - no resampling, true - resampling
%   plotFunc    Handle for plot function that is called when a filter
%               recursion has finished.
% Output:
%   xfp         [n x K] Posterior means of particle filter
%   Pfp         [n x n x K] Posterior error covariances of particle filter
%   Xp          [n x N x K] Non-resampled Particles for posterior state distribution in times 1:K
%   Wp          [N x K] Non-resampled weights for posterior state x in times 1:K

% Your code here, please. 
n = size(x_0,1);m = size(Y,1); K = size(Y,2);
xfp = zeros(n,K);
Pfp = zeros(n,n,K);
Xp = zeros(n,N,K);
Wp = zeros(N,K);%have to be transposed latter
storage=zeros(n,N,K);
if size(x_0,2) == 1
    X_kmin1 = mvnrnd(x_0,P_0,N)';
end
W_kmin1=ones(N,1)/N;
    for i=1:K
        W_kmin1=W_kmin1';
        %resampling
        if bResample
            [Xk, Wk, j] = resampl(X_kmin1,W_kmin1);%X=[n x N]
        else
            Xk=X_kmin1;
            Wk=W_kmin1;
            j=[1:1:N];
        end
        %predict
        [X_kmin1,W_kmin1] = pfFilterStep(Xk, Wk, Y(:,i), proc_f, proc_Q, meas_h, meas_R);%[n x N]
        %store it
        Xp(:,:,i)=X_kmin1;
        Wp(:,i)=W_kmin1';
        if ~isempty(isOnRoad)
            u = isOnRoad(Xp(1,:,i),Xp(2,:,i));
            Wp(:,i) = Wp(:,i) .* u;
            Wp(:,i) = Wp(:,i)/sum(Wp(:,i));
            W_kmin1=Wp(:,i)';
        end
        % storage(:,:,i)=Xp(:,:,i).*Wp(:,i)';
        xfp(:,i)=sum(Xp(:,:,i).*Wp(:,i)',2);
        Pfp(:,:,i)=Wp(:,i)'.*(Xp(:,:,i) - xfp(:,i))*(Xp(:,:,i) - xfp(:,i))';
        if ~isempty(plotFunc)&&i>=2
            plotFunc(i, Xp(:,:,i), Xp(:,:,i-1), Wp(:,i)', j);
        end

    % If you want to be a bit fancy, then only store and output the particles if the function
    % is called with more than 2 output arguments.
    end
end

function [Xr, Wr, j] = resampl(X, W)
    % Copy your code from previous task! 
    %RESAMPLE Resample particles and output new particles and weights.
% resampled particles. 
%
%   if old particle vector is x, new particles x_new is computed as x(:,j)
%
% Input:
%   X   [n x N] Particles, each column is a particle.
%   W   [1 x N] Weights, corresponding to the samples
%
% Output:
%   Xr  [n x N] Resampled particles, each corresponding to some particle 
%               from old weights.
%   Wr  [1 x N] New weights for the resampled particles.
%   j   [1 x N] vector of indices refering to vector of old particles

    % Your code here!
    n=size(X,1);
    N=size(X,2);
    W=W/sum(W);
    Wr=zeros(1,N);
    Xr=zeros(n,N);
    c=zeros(1,N);
    c(1)=W(1);
    j=zeros(1,N);
    % generate c
    for i=2:N
        c(i)=c(i-1)+W(i);
    end
    %generate random value to see which domain it belongs to
    for i =1:N
       a=rand;%uniform distribution
       for k=1:N
           if a<c(k)
               Xr(:,i)=X(:,k);
               j(i)=k;
               break;
           end
       end
    end
    Wr=ones(1,N)*(1/N);

end

function [X_k, W_k] = pfFilterStep(X_kmin1, W_kmin1, yk, proc_f, proc_Q, meas_h, meas_R)
        % Copy your code from previous task!
        %PFFILTERSTEP Compute one filter step of a SIS/SIR particle filter.
    %
    % Input:
    %   X_kmin1     [n x N] Particles for state x in time k-1
    %   W_kmin1     [1 x N] Weights for state x in time k-1
    %   y_k         [m x 1] Measurement vector for time k
    %   proc_f      Handle for process function f(x_k-1)
    %   proc_Q      [n x n] process noise covariance
    %   meas_h      Handle for measurement model function h(x_k)
    %   meas_R      [m x m] measurement noise covariance
    %
    % Output:
    %   X_k         [n x N] Particles for state x in time k
    %   W_k         [1 x N] Weights for state x in time k
    n=size(X_kmin1,1);
    N=size(X_kmin1,2);
    W_k=zeros(1,N);
    X_k=zeros(n,N);
    % Your code here!
    for i=1:N
        X_k(:,i)=proc_f(X_kmin1(:,i))+mvnrnd(zeros(n,1),proc_Q)';
    end
    for i=1:N
        W_k(i) = W_kmin1(i) * mvnpdf(yk, meas_h(X_k(:, i)), meas_R);
    end
    W_k=W_k/sum(W_k);
end