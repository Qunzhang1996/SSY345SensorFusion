%% Question1

%(a)
%X_k=A*X_{k-1}+q_{k-1}, q_{k-1}~N(0,Q)
% Y_k=H*X_k+r_k, r_k~N(0,R)
Q=1.5;R=3;A=1;H=1;
x_0=2;P_0=8;N=36;
X_k1= genLinearStateSequence(x_0, P_0, A, Q, N);
Y_k1= genLinearMeasurementSequence(X_k1, H, R);
figure('Position',[300 300 600 400]);hold on;
plot(0:N-1,X_k1(1,1:N), '--k','Color','red',LineWidth=1);
plot(1:N-1, Y_k1(1,1:N-1),'Color','blue',LineWidth=1);
grid on
legend('State sequence', 'Measurements')
xlabel('k');
ylabel('value of X_{k} and Y_{k}');
hold off
%%
%(b)
[X_estimate, P_estimate,~,~,~] = kalmanFilter(Y_k1, x_0, P_0, A, Q, H, R);
X_true =X_k1;
x_3sigma1=X_estimate+3*sqrt(P_estimate);
x_3sigma2=X_estimate-3*sqrt(P_estimate);
figure('Position',[300 300 600 400]);hold on;
plot(0:N-1,X_true(1,1:N), '--k','Color','red',LineWidth=1);
plot(1:N-1, Y_k1(1,1:N-1),'Color','blue',LineWidth=1);
plot(1:N-1, X_estimate(1,1:N-1),'Color','green',LineWidth=2);
plot(1:N-1, x_3sigma1(1,1:N-1), '--k','Color','m',LineWidth=1);
plot(1:N-1, x_3sigma2(1,1:N-1), '--k','Color','cyan',LineWidth=1);
grid on
legend('True states', 'Measurements','$\hat{X_k}$','$\hat{X_k}+3\sigma$','$\hat{X_k}-3\sigma$','Interpreter','latex')
xlabel('k');
ylabel('values');
hold off
%%
mu=0;value=10;K=[1,2,4,30];
Y_densty=[];
for i=1:length(K)
    % variance=(X_true(:,K(i)+1)-X_estimate(:,K(i)))^2;
    variance=P_estimate(i)
    [x_plot,Y1]=normPlot(value,mu,variance);
    Y_densty=[Y_densty;Y1];
end
figure('Position',[300 300 600 400]);hold on;
plot(x_plot,Y_densty(1,:),'LineWidth',1,'color','blue');
plot(x_plot,Y_densty(2,:),'LineWidth',1,'color','red');
plot(x_plot,Y_densty(3,:),'LineWidth',1,'color','cyan');
plot(x_plot,Y_densty(4,:),'LineWidth',1,'color','green');
legend('K=1', 'K=2','K=4','K=30');
grid on
xlabel('x');
ylabel('values');
hold off
%%
%(c)
X_NEW=12;
[X_estimate2, ~,~,~,~] = kalmanFilter(Y_k1, X_NEW, P_0, A, Q, H, R);
figure('Position',[300 300 600 400]);hold on;
plot(0:N-1,X_true(1,1:N), '--k','Color','blue',LineWidth=1);
plot(1:N-1, X_estimate(1,1:N-1),'Color','green',LineWidth=2);
plot(1:N-1, X_estimate2(1,1:N-1),'--k','Color','red',LineWidth=2);
legend('True states','$X_{estimate1}$', '$X_{estimate2}$','Interpreter','latex');
grid on
xlabel('x');
ylabel('values');
hold off
%%
%(d)
%X_k=A*X_{k-1}+q_{k-1}, q_{k-1}~N(0,Q)
% Y_k=H*X_k+r_k, r_k~N(0,R)
%Q=1.5;R=3;A=1;H=1;
%x_0=2;P_0=8;N=35;P(x0)=N(x0;x_0,P_0)
%let K=34
k=20;value=8;
[X_estimate, P_estimate, X_predict, P_predict,~] = kalmanFilter(Y_k1, X_NEW, P_0, A, Q, H, R);
[x_plot1,X1]=normPlot(value,X_estimate(:,k-1),P_estimate(:,k-1));
[x_plot2,X2]=normPlot(value,X_predict(:,k),P_predict(:,k-1));
[x_plot3,X3]=normPlot(value,X_estimate(:,k),P_estimate(:,k));
[x_plot4,Y_k]=normPlot(value,X_k1(:,k+1),R);
figure('Position',[300 300 600 400]);hold on;
plot(x_plot1,X1,'--k','LineWidth',1,'color','blue');
plot(x_plot2,X2,'LineWidth',1,'color','red');
plot(x_plot3,X3,'LineWidth',1,'color','cyan');
xline(Y_k1(:,k),'LineWidth',1,'color','green');
legend('$P(X_{k-1|k-1})$','$P(X_{k|k-1})$','$P(X_{k|k})$','$y_k$','Interpreter','latex');
grid on
xlabel('x');
ylabel('values');
hold off
%%
%(e)
%question 1
Q=1.5;R=3;A=1;H=1;
x_0=2;P_0=8;N2=10000;
X_true2= genLinearStateSequence(x_0, P_0, A, Q, N2);
Y_k2= genLinearMeasurementSequence(X_true2, H, R);
[X_estimate3, P_estimate3, ~, ~,~] = kalmanFilter(Y_k2, x_0, P_0, A, Q, H, R);
delta_X=X_true2(:,2:end)-X_estimate3;
figure('Position',[300 300 600 400]);hold on;
histogram(delta_X,'Normalization','pdf','FaceColor',[.9 .9 .9]);
value=8;mu=0;
[x_plote,Ye]=normPlot(value,mu,P_estimate3(:,N2));
plot(x_plote,Ye,'--k','LineWidth',2,'color','blue');
legend('$X_k-\hat{X_{k|k}}$','$N(x;0,P_{N|N})$','Interpreter','latex');
grid on
xlabel('x');
ylabel('values');
hold off
%%
%question 2
[X_estimate3, P_estimate3, ~, Pred_P3,VK] = kalmanFilter(Y_k2, x_0, P_0, A, Q, H, R);
autocorr(VK)
%%
s=H*Pred_P3(length(Y_k2))*H'+R;
figure('Position',[300 300 600 400]);hold on;
plot(1:length(Y_k2),s^(-1/2)*VK);
yline(3*sqrt(P_estimate3(length(Y_k2))))
yline(-3*sqrt(P_estimate3(length(Y_k2))))
xlabel('x');
ylabel('values');
hold off
%% Question2

%2(a)
value=8;
%ypk=pk+rpk, rpk~N(0,1),measured position
%yvk=C(vk+rvk),C≈1,rvk~N(0,R)
load SensorMeasurements.mat
% [mu_y, Sigma_y] = approxGaussianTransform(CalibrationSequenceVelocity_v10)
%% Data of histogram

[mu_y1, Sigma_y1] = approxGaussianTransform(CalibrationSequenceVelocity_v0)
figure('Position',[300 300 600 400]);hold on;
histogram(CalibrationSequenceVelocity_v0,'Normalization','pdf','FaceColor',[.9 .9 .9]);
[x_plote,Ye]=normPlot(value,mu_y1,Sigma_y1);
plot(x_plote,Ye,'--k','LineWidth',2,'color','blue');
legend('$Velocity_{v}$','$analytical$','Interpreter','latex');
grid on
xlabel('x');
ylabel('values');
hold off
[mu_y2, Sigma_y2,C2]=C_calculate(CalibrationSequenceVelocity_v10,10)
figure('Position',[300 300 600 400]);hold on;
histogram(CalibrationSequenceVelocity_v10,'Normalization','pdf','FaceColor',[.9 .9 .9]);
[x_plote,Ye2]=normPlot(value,mu_y2, Sigma_y2);
plot(x_plote,Ye2,'--k','LineWidth',2,'color','blue');
legend('$Velocity_{v}$','$analytical$','Interpreter','latex');
grid on
xlabel('x');
ylabel('values');
hold off
[mu_y3, Sigma_y3,C3]=C_calculate(CalibrationSequenceVelocity_v20,20)
figure('Position',[300 300 600 400]);hold on;
histogram(CalibrationSequenceVelocity_v20,'Normalization','pdf','FaceColor',[.9 .9 .9]);
[x_plote,Ye3]=normPlot(value,mu_y3, Sigma_y3);
plot(x_plote,Ye3,'--k','LineWidth',2,'color','blue');
legend('$Velocity_{v}$','$analytical$','Interpreter','latex');
grid on
xlabel('x');
ylabel('values');
hold off
R=(Sigma_y1/C3^2+Sigma_y2/C3^2+Sigma_y3/C3^2)/3



figure('Position',[300 300 600 400]);hold on;
histogram([CalibrationSequenceVelocity_v0-mu_y1,CalibrationSequenceVelocity_v10-mu_y2,CalibrationSequenceVelocity_v20-mu_y3],'Normalization','pdf','FaceColor',[.9 .9 .9]);
[x_plote,Ye4]=normPlot(value,0 ,R);
plot(x_plote,Ye4,'--k','LineWidth',2,'color','blue');
legend('$Velocity_{v}$','$analytical$','Interpreter','latex');
grid on
xlabel('x');
ylabel('values');
hold off
%%
%2(b)
SensorData=Generate_y_seq()
%%%%%data process also implemented in KF
k=length(SensorData);
newData=[];
for i=1:k
    if ~isnan(SensorData(:,i))%mod(i,2)~=0
        newData=[newData,SensorData(:,i)];
    end
end
newData

SensorData1=SensorData(:,1:2:end);
%%
%2(c)
%%%changing Q, cv and ca will be different!
%%%witness can also be shown to convince it
T=0.2;N=1000;
%Constant Velocity Model
A_cv=[1 T;0 1];H=[1 0;0 C3];R1=[1 0;0 C3^2*R];Q=[0 0;0 1];x_0=[1;1];P_0=100*eye(2);
[X_cv, P_cv, ~, P_precv,VK_cv] = kalmanFilter(SensorData, x_0, P_0, A_cv, Q, H, R1);
X_states = genLinearStateSequence(x_0, P_0, A_cv, Q, N);
%Time Constant Acceleratiion Model
A_ca=[1 T T^2/2;0 1 T;0 0 1];H_ca=[1 0 0;0 C3 0];Q_ca=[0 0 0;0 0 0;0 0 1];x_0=[1;1;1];P_0=100*eye(3);
R2=[1 0;0 C3^2*R];
[X_ca, P_ca, ~, P_preca,VK_ca] = kalmanFilter(SensorData, x_0, P_0, A_ca, Q_ca, H_ca, R2);
X_states_ca = genLinearStateSequence(x_0, P_0, A_ca, Q_ca, N);


figure('Position',[300 300 600 400]);hold on;
plot(1:length(X_cv),X_cv(1,:),'--k','LineWidth',4,'color','blue');
% plot(1:length(X_cv),X_ca(1,:),'o','LineWidth',2,'color','red');
plot(1:length(X_cv),newData(1,:),'LineWidth',2,'color','green');
legend('$position_{cv}$','$measurement$','Interpreter','latex');
% legend('$position_{ca}$','$measurement$','Interpreter','latex');
% legend('$position_{cv}$','$position_{ca}$','$measurement$','Interpreter','latex');
grid on
xlabel('time');
ylabel('values');
hold off
figure('Position',[300 300 600 400]);hold on;
% plot(1:length(X_cv),X_cv(1,:),'--k','LineWidth',4,'color','blue');
plot(1:length(X_cv),X_ca(1,:),'o','LineWidth',2,'color','red');
plot(1:length(X_cv),newData(1,:),'LineWidth',2,'color','green');
% legend('$position_{cv}$','$measurement$','Interpreter','latex');
legend('$position_{ca}$','$measurement$','Interpreter','latex');
% legend('$position_{cv}$','$position_{ca}$','$measurement$','Interpreter','latex');
grid on
xlabel('time');
ylabel('values');
hold off
figure('Position',[300 300 600 400]);hold on;
% plot(1:length(X_cv),X_states(2,2:end),'--k','LineWidth',4,'color','cyan');
plot(1:length(X_cv),X_cv(2,:),'--k','LineWidth',4,'color','blue');
% plot(1:length(X_cv),X_states(1,1:N),'LineWidth',2,'color','red');
plot(1:length(X_cv),newData(2,:)/C3,'LineWidth',2,'color','green');
legend('$velocity_{cv}$','$measurement$','Interpreter','latex');
grid on
xlabel('time');
ylabel('values');
hold off
figure('Position',[300 300 600 400]);hold on;
% plot(1:length(X_cv),X_states(2,2:end),'--k','LineWidth',4,'color','cyan');
% plot(1:length(X_cv),X_states(1,1:N),'LineWidth',2,'color','red');
plot(1:length(X_cv),X_ca(2,:),'LineWidth',2,'color','red');
plot(1:length(X_cv),newData(2,:)/C3,'LineWidth',1,'color','green');
legend('$velocity_{ca}$','$measurement$','Interpreter','latex');
grid on
xlabel('time');
ylabel('values');
hold off
%%
autocorr(VK_cv(1,:))
% [X_cv, P_cv, ~, P_precv,VK_cv] = kalmanFilter(SensorData, x_0, P_0, A_cv, Q, H, R1);
s1=H*P_precv(:,:,1000)*H'+R1;
figure('Position',[300 300 600 400]);hold on;
S1=s1^(-1/2)*VK_cv
plot(S1(1,:));
yline(3*sqrt(P_cv(1,length(SensorData))))
yline(-3*sqrt(P_cv(1,length(SensorData))))
xlabel('x');
ylabel('values');
hold off
%%
autocorr(VK_cv(2,:))
figure('Position',[300 300 600 400]);hold on;
S1=s1^(-1/2)*VK_cv
plot(S1(2,:));
yline(3*sqrt(P_cv(2,length(SensorData))))
yline(-3*sqrt(P_cv(2,length(SensorData))))
xlabel('x');
ylabel('values');
hold off
%%
% [X_ca, P_ca, ~, P_preca,VK_ca] = kalmanFilter(SensorData, x_0, P_0, A_ca, Q_ca, H_ca, R2);
s2=H_ca*P_ca(:,:,1000)*H_ca'+R2;
autocorr(VK_ca(1,:))
figure('Position',[300 300 600 400]);hold on;
S2=s2^(-1/2)*VK_ca
plot(S1(1,:));
yline(3*sqrt(P_ca(1,length(SensorData))))
yline(-3*sqrt(P_ca(1,length(SensorData))))
xlabel('x');
ylabel('values');
hold off
%%
autocorr(VK_ca(2,:))
figure('Position',[300 300 600 400]);hold on;
S2=s2^(-1/2)*VK_ca
plot(S1(2,:));
yline(3*sqrt(P_ca(2,length(SensorData))))
yline(-3*sqrt(P_ca(2,length(SensorData))))
xlabel('x');
ylabel('values');
hold off

% 
% figure('Position',[300 300 600 400]);hold on;
% plot(1:length(X_cv),X_ca(1,:),'--k','LineWidth',2,'color','blue');
% plot(1:length(X_cv),X_states_ca(1,1:N),'LineWidth',2,'color','red');
% legend('$position_{kf}$','$position_s$','Interpreter','latex');
% grid on
% xlabel('time');
% ylabel('values');
% hold off
% figure('Position',[300 300 600 400]);hold on;
% plot(1:length(X_cv),X_ca(2,:),'--k','LineWidth',2,'color','blue');
% plot(1:length(X_cv),X_states_ca(1,1:N),'LineWidth',2,'color','red');
% legend('$velocity_{kf}$','$velocity_s$','Interpreter','latex');
% grid on
% xlabel('time');
% ylabel('values');
% hold off
%Input:
%   Y           [m x N] Measurement sequence
%   x_0         [n x 1] Prior mean
%   P_0         [n x n] Prior covariance
%   A           [n x n] State transition matrix
%   Q           [n x n] Process noise covariance
%   H           [m x n] Measurement model matrix
%   R           [m x m] Measurement noise covariance
%

%% Function of approxGaussian

function [mu_y, Sigma_y,C]=C_calculate(data,vk)
[mu_y, Sigma_y] = approxGaussianTransform(data);
C=mu_y/vk;
end
function [mu_y, Sigma_y] = approxGaussianTransform(data)
N=length(data);
mu_y=mean(data,2);
Sigma_y=(data-mu_y)*(data-mu_y)'/(N-1);
end
%% Function of normpdf

function [x,Y]=normPlot(value,mu,variance)
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
x=[mu-value:0.1:mu+value];
Y=normpdf(x,mu,sqrt(variance));
end
%% function of kalman filter

function X = genLinearStateSequence(x_0, P_0, A, Q, N)
%GENLINEARSTATESEQUENCE generates an N-long sequence of states using a 
%    Gaussian prior and a linear Gaussian process model
%
%Input:
%   x_0         [n x 1] Prior mean
%   P_0         [n x n] Prior covariance
%   A           [n x n] State transition matrix
%   Q           [n x n] Process noise covariance
%   N           [1 x 1] Number of states to generate
%
%Output:
%   X           [n x N+1] State vector sequence
%

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
X0= mvnrnd(x_0, P_0)';
% disp(size(X0))
% disp(X0)
X=X0;
for i=1:N
    q=mvnrnd(zeros(length(x_0),1), Q)';
    X=[X A*X(:,i)+q];
end
% disp(X)
% Your code here
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
end
function Y = genLinearMeasurementSequence(X, H, R)
%GENLINEARMEASUREMENTSEQUENCE generates a sequence of observations of the state 
% sequence X using a linear measurement model. Measurement noise is assumed to be 
% zero mean and Gaussian.
%
%Input:
%   X           [n x N+1] State vector sequence. The k:th state vector is X(:,k+1)
%   H           [m x n] Measurement matrix
%   R           [m x m] Measurement noise covariance
%
%Output:
%   Y           [m x N] Measurement sequence
%
Y=[];
for i=2:length(X)
    Y=[Y, H*X(:,i)+mvnrnd(zeros(length(R),1),R)'];
end
% your code here
end
function [x, P] = linearPrediction(x, P, A, Q)
%LINEARPREDICTION calculates mean and covariance of predicted state
%   density using a liear Gaussian model.
%
%Input:
%   x           [n x 1] Prior mean
%   P           [n x n] Prior covariance
%   A           [n x n] State transition matrix
%   Q           [n x n] Process noise covariance
%
%Output:
%   x           [n x 1] predicted state mean
%   P           [n x n] predicted state covariance
%
x=A*x;
P=A*P*A'+Q;
% Your code here
end
function [x, P,VK] = linearUpdate(x, P, y, H, R)
%LINEARPREDICTION calculates mean and covariance of predicted state
%   density using a linear Gaussian model.
%
%Input:
%   x           [n x 1] Prior mean
%   P           [n x n] Prior covariance
%   y           [m x 1] Measurement
%   H           [m x n] Measurement model matrix
%   R           [m x m] Measurement noise covariance
%
%Output:
%   x           [n x 1] updated state mean
%   P           [n x n] updated state covariance
%
SK=H*P*H'+R;
KK=P*H'*inv(SK);
VK=y-H*x;
x=x+KK*VK;
P=P-KK*SK*KK';
% Your code here

end
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