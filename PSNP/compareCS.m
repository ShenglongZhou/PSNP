% demon  compressed sensing problems 
clc; close all; clear all; warning off; 
addpath(genpath(pwd));

n     = 10000; 
m     = ceil(0.25*n);
s     = ceil(0.05*n);     

T       = randperm(n,s);  
xopt    = zeros(n,1);  
xopt(T) = (0.5+1*rand(s,1)).*(2*randi([0,1],[s,1])-1);  
data.A  = normalization(randn(m,n), 3); 
data.b  = data.A(:,T)*xopt(T)+ 0.00*randn(m,1);  

pars1.prob  = 'CS';  pars2.prob  = 'CS';
pars1.cond  = 1;     pars2.cond  = 0;

q0          = [0 1/4 1/2 2/3 3/4];
lam         = 0.025*norm(data.b'*data.A,'inf');
out1        = cell(length(q0),1);
out2        = cell(length(q0),1);
for i       = 1:length(q0)  
    func    = @(x,T,key)funCS(x,T,key,data);
    out1{i} = PSNP(func,n,lam,q0(i),pars1);   
    out2{i} = PSNP(func,n,lam,q0(i),pars2); 
end

fprintf('\n    q      Objective    Accuracy    Sparsity    Time(sec)')
fprintf('      Objective    Accuracy    Sparsity    Time(sec)\n');
fprintf(' -------   ---------------------PCSNP--------------------');
fprintf('      ----------------------PSNP--------------------\n');
for i   = 1:length(q0) 
    acc = @(v)norm(v-xopt)/norm(xopt);
    fprintf(' %6.3f    %5.2e     %5.2e     %4d      %7.3f',...
    q0(i), out1{i}.obj, acc(out1{i}.sol), nnz(out1{i}.sol), out1{i}.time)
    fprintf('%17.2e     %5.2e     %4d    %9.3f   \n', ...
    out2{i}.obj, acc(out2{i}.sol),nnz(out2{i}.sol), out2{i}.time);
end
