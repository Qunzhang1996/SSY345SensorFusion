function plotbycycle(X, Y, xfp, Pfp,Xp, PLT)
plot([1+i 1+9*i 5+9*i])
plot([7+9*i 11+9*i 11+i 7+i]);plot([5+i 1+i])
plot([2+5.2*i 2+8.3*i 4+8.3*i 4+5.2*i 2+5.2*i])%House 1
plot([2+3.7*i 2+4.4*i 4+4.4*i 4+3.7*i 2+3.7*i])%House 2
plot([2+2*i 2+3.2*i 4+3.2*i 4+2*i 2+2*i])%House 3
plot([5+i 5+2.2*i 7+2.2*i 7+i])%House 4
plot([5+2.8*i 5+5.5*i 7+5.5*i 7+2.8*i 5+2.8*i])%House 5
plot([5+6.2*i 5+9*i]);plot([7+9*i 7+6.2*i 5+6.2*i])%House 6
plot([8+4.6*i 8+8.4*i 10+8.4*i 10+4.6*i 8+4.6*i])%House 7
plot([8+2.4*i 8+4*i 10+4*i 10+2.4*i 8+2.4*i])%House 8
plot([8+1.7*i 8+1.8*i 10+1.8*i 10+1.7*i 8+1.7*i])%House 9

% axis([0.8 11.2 0.8 9.2])
title('A map of the village','FontSize',20)
plot([X+Y*1i],'-*')
plot([xfp(1,:)+xfp(2,:)*1i],'-*','Color','cyan',LineWidth=1);
N=length(Y);
    if ~isempty(PLT)
        for m=1:10:N
            u=isOnRoad(Xp(1,:,m),Xp(2,:,m));
            x_test=reshape(Xp(1,:,m),[length(Xp(1,:,m)),1]);
            y_test=reshape(Xp(2,:,m),[length(Xp(2,:,m)),1]);
            covEllipse = sigmaEllipse2D(xfp(1:2,m),Pfp(1:2,1:2,m),3,100);
            plot(covEllipse(1,:),covEllipse(2,:),'black','LineWidth',1);
            fill(covEllipse(1,:),covEllipse(2,:), 'cyan','FaceAlpha', 0.2);
            scatter(x_test.*u,y_test.*u,5);
        end
    end
end
