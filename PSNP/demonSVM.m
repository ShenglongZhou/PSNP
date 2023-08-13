% demon sparse support vector machine problems 
clc; close all; warning off; addpath(genpath(pwd));

dat     = load('arcene.mat'); 
class   = load('arceneclass.mat');  
b       = class.y;
b( b   ~= 1) = -1;
data.A  = [normalization(dat.X,2)  ones(size(b))];
data.b  = b;
[m,n]   = size(data.A); 
data.At = data.A';
 
pars.prob  = 'SVM';
pars.tol   = 1e-5*log2(m*n);
q0         = [0 1/2 2/3];
lam        = @(q)3e-4*(1+6*q)*log2(n/m)*norm(data.b'*data.A,'inf')/m;
for i      = 1:length(q0) 
    lambda = lam(q0(i));
    func   = @(xT,T,key)funSVM(xT,T,key,lambda,data);
    out{i} = PSNP(func,n,lambda,q0(i),pars);   
end

acc = @(v)( 1-nnz(data.b - sign( data.A(:,v~=0)*v(v~=0) ))/m );
fprintf('   q      Objective     Accuracy     Time(sec)\n');
fprintf(' ---------------------------------------------\n');
for i  = 1:length(q0)
    fprintf('%6.3f     %5.2e      %5.2f%%      %7.3f    \n', ...
    q0(i),out{i}.obj,acc(out{i}.sol)*100,out{i}.time);
end
