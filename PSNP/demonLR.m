% demon sparse logistic regression problems 
clc; close all; warning off; addpath(genpath(pwd));
 
dat     = load('arcene.mat'); 
class   = load('arceneclass.mat');  
b       = class.y;
b( b   ~= 1) = 0;
[m,n]   = size(dat.X);   
data.A  = normalization(dat.X,2);
data.b  = b;
data.At = data.A';

pars.prob  = 'LR';
q0         = [0 1/2 2/3];
lam        = @(q)1e-3*(1+6*q)*norm(data.b'*data.A,'inf')/m;
for i      = 1:length(q0) 
    lambda = lam(q0(i));
    func   = @(xT,T,key)funLR(xT,T,key,lambda,data);
    out{i} = PSNP(func,n,lambda,q0(i),pars);   
end

acc = @(v)( 1-nnz(data.b - sign( max( data.A(:,v~=0)*v(v~=0),0 ) ))/m );
fprintf('   q     LogistLoss   Accuracy   CPUtime\n');
fprintf(' ---------------------------------------\n');
for i  = 1:length(q0)
    fprintf('%6.3f    %5.2e    %5.2f%%     %5.3f    \n', ...
    q0(i),out{i}.obj,acc(out{i}.sol)*100,out{i}.time);
end
