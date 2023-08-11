% demon  compressed sensing problems 
clc; close all; warning off; addpath(genpath(pwd));

n     = 10000; 
m     = ceil(0.25*n);
s     = ceil(0.025*n);     

nf      = 0.025;
T       = randperm(n,s);  
xopt    = zeros(n,1);  
xopt(T) = (0.5+1*rand(s,1)).*(2*randi([0,1],[s,1])-1);  
data.A  = normalization(randn(m,n), 3); 
data.b  = data.A(:,T)*xopt(T)+ nf*randn(m,1);  

pars.prob  = 'CS';
q0         = [0 1/2 2/3];
lam        = @(q)0.02*(1+3*q)*norm(data.b'*data.A,'inf');
for i      = 1:length(q0) 
    lambda = lam(q0(i));
    func   = @(x,T,key)funCS(x,T,key,data);
    out{i} = PSNP(func,n,lambda,q0(i),pars);   
end

fprintf('   q     Objective   Accuracy   CPUtime\n');
fprintf(' --------------------------------------\n');
for i      = 1:length(q0)
    fprintf('%6.3f   %5.2e    %5.2e    %.3f    \n', ...
    q0(i),out{i}.obj,norm(out{i}.sol-xopt)/norm(xopt),out{i}.time);
end
