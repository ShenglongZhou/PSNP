% demon sparse support vector machine problems 
clc; close all; clear all; warning off; 
addpath(genpath(pwd));

dat     = load('arcene.mat'); 
class   = load('arceneclass.mat');  
b       = class.y;
b( b   ~= 1) = -1;
data.A  = [normalization(dat.X,1)  ones(size(b))];
data.b  = b;
[m,n]   = size(data.A); 
  
pars.prob = 'SVM';
pars.cond = 0;
pars.x0   = [zeros(n-1,1);1];
q         = 0;
lam       = 0.4*norm(data.b'*data.A,'inf')/m/log10(10+n);
func      = @(xT,T,key)funSVM(xT,T,key,20/m,data);
out       = PSNP(func,n,lam,q,pars);   

acc = @(v)( 1-nnz(data.b - sign( data.A(:,v~=0)*v(v~=0) ))/m );
fprintf('\n    q      Objective     Accuracy      Sparsity     Time(sec)\n');
fprintf(' -------------------------------------------------------------\n');
fprintf(' %6.3f    %5.2e      %5.2f%%      %6d       %7.3f    \n', ...
    q,out.obj,acc(out.sol)*100, nnz(out.sol), out.time);
