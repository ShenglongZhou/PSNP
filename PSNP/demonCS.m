% demon  compressed sensing problems  
clc; close all; clear all; warning off; 
addpath(genpath(pwd));

n     = 2000; 
m     = ceil(0.25*n);
s     = ceil(0.05*n);     

T       = randperm(n,s);  
xopt    = zeros(n,1);  
xopt(T) = (0.5+1*rand(s,1)).*(2*randi([0,1],[s,1])-1);  
data.A  = normalization(randn(m,n), 3); 
data.b  = data.A(:,T)*xopt(T)+ 0.00*randn(m,1);  

pars.prob  = 'CS';  
pars.cond  = 0;   
q          = 0;
lam        = 0.025*norm(data.b'*data.A,'inf');
func       = @(x,T,key)funCS(x,T,key,data);
out        = PSNP(func,n,lam,q,pars);  
recoverShow(xopt,out.sol,[900,500,500,250],1)
