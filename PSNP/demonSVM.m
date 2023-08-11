% demon sparse logistic regression problems 
clc; close all; clear all; warning off
addpath(genpath(pwd));
 
dat    = load('arcene.mat'); 
class  = load('arceneclass.mat');  

b      = class.y;
b( b  ~= 1) = -1;
data.A = [normalization(dat.X,2)  ones(size(b))];
data.b = b;
[m,n]  = size(data.A); 
    
pars.prob  ='SVM';
pars.tol   = 1e-8*log2(m*n);
q          = 0; 
lambda     = 3e-4*log2(n/m)*norm(data.b'*data.A,'inf')/m;
func       = @(xT,T,key)funSVM(xT,T,key,lambda,data);
out        = PSNP(func,n,lambda,q,pars);

acc = @(v)( 1-nnz(data.b - sign( data.A(:,v~=0)*v(v~=0) ))/m );
fprintf(' Sample size:           %d x %d\n', m,n);
fprintf(' Logistic Loss:         %5.2e\n', out.obj);
fprintf(' Classification error:  %5.3f%%\n', acc(out.sol)*100);
fprintf(' CPU time:              %.3fsec\n',  out.time);