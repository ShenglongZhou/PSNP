% demon sparse logistic regression problems 
clc; close all; clear all; warning off
addpath(genpath(pwd));
 
dat    = load('arcene.mat'); 
class  = load('arceneclass.mat');  

b      = class.y;
b( b  ~= 1) = 0;
[m,n]  = size(dat.X);  
ntp    = 2; 
data.A = normalization(dat.X,ntp);
data.b = b;

pars.prob   ='LR';
q           = 0; 
lambda      = 1e-3*norm(data.b'*data.A,'inf')/m;
func        = @(xT,T,key)funLR(xT,T,key,2*lambda,data);
out         = PNPLq(func,n,lambda,q,pars);

acc = @(v)( mean(abs(data.b - sign( max( data.A(:,v~=0)*v(v~=0),0 ) )) ) );
fprintf(' Sample size:           %d x %d\n', m,n);
fprintf(' Logistic Loss:         %5.2e\n', out.obj);
fprintf(' Classification error:  %5.3f%%\n', acc(out.sol)*100);
fprintf(' CPU time:              %.3fsec\n',  out.time);