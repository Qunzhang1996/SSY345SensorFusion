function plothistogram(data,mu,sigma,signal)
figure('Position',[300 300 600 400]);hold on;
for i=1:3
    subplot(3,1,i);hold on
    histogram(data(i,:),'Normalization','pdf','FaceColor',[.9 .9 .9]);
    [x_plote,Ye1]=normPlot(mu(i,:),sigma(i,i));
    plot(x_plote,Ye1,'LineWidth',2);hold off
    legend('histogram','normpdf')
end
figure('Position',[300 300 600 400]);hold on;
for j=1:3
    subplot(3,1,j);
    plot(signal(j,:),'LineWidth',1);
    legend('signal')
end
end