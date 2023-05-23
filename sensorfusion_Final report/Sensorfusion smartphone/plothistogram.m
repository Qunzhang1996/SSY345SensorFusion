function plothistogram(data,mu,sigma,signal)
figure('Position',[300 300 600 400]);hold on;
list={'x','y','z'};
for i=1:3
    subplot(3,1,i);hold on
    histogram(data(i,:),'Normalization','pdf','FaceColor',[.9 .9 .9]);
    % [x_plote,Ye1]=normPlot(mu(i,:),sigma(i,i));
    % plot(x_plote,Ye1,'LineWidth',2);hold off
    title('histogram of axis',list(i))
    legend('histogram')
end
figure('Position',[300 300 600 400]);hold on;
for j=1:3
    subplot(3,1,j);
    plot(signal(j,:),'LineWidth',1);
    legend('signal')
    title('signal of axis',list{j})
end
end