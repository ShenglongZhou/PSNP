% demon  compressed sensing problems 
 clc; close all; clear all; warning off
 addpath(genpath(pwd));

n     = 10000; 
m     = ceil(0.25*n);
s     = ceil(0.025*n);     

nf      = 0.025;
T       = randperm(n,s);  
xopt    = zeros(n,1);  
xopt(T) = (0.5+1*rand(s,1)).*(2*randi([0,1],[s,1])-1);  
data.A  = normalization(randn(m,n), 3); 
data.b  = data.A(:,T)*xopt(T)+ nf*randn(m,1);  

pars.prob = 'CS';
q         = 0/3;
lambda    = 0.02*norm(data.b'*data.A,'inf'); 

func      = @(x,T,key)funCS(x,T,key,data);
out       = PNPLq(func,n,lambda,q,pars); 

fprintf(' Sample size:           %d x %d\n', m,n);
fprintf(' Objective:             %5.2e\n',   out.obj); 
fprintf(' Recovery accuracy:     %5.2e\n',   norm(out.sol-xopt)/norm(xopt)); 
fprintf(' CPU time:              %.3fsec\n',  out.time);