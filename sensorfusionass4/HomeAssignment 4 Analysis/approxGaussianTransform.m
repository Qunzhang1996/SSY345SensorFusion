function [mu_y, Sigma_y] = approxGaussianTransform(data)
N=length(data);
mu_y=mean(data,2);
Sigma_y=(data-mu_y)*(data-mu_y)'/(N-1);
end