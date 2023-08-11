function Out = PSNP(funcF,n,lambda,q,pars)
% This code aims at solving the Lq norm regularized optimization with form
%
%         min_{x\in R^n} F(x) := f(x) + \lambda \|x\|_q^q,
%
% where \lambda>0 and q\in[0,1).  
%--------------------------------------------------------------------------
% Inputs:
%     func:  A function handle defines (objective,gradient,sub-Hessain) of F (required)
%     n   :  Dimension of the solution x                                     (required)
%     lambda : Penalty parameter                                             (required) 
%     q      : one of {0 1/2, 2/3}                                           (required)
%     pars:  Parameters are all OPTIONAL
%            pars.x0      --  Starting point of x,   pars.x0=zeros(n,1) (default)
%            pars.prob    --  ='CS', 'SVM, or 'LR'  
%            pars.show    --  Display results or not for each iteration (default, 1)
%            pars.maxit   --  Maximum number of iterations, (default,2000) 
%            pars.tol     --  Tolerance of the halting condition, (default,1e-6)
%
% Outputs:
%     Out.sol :   The sparse solution x
%     Out.time:   CPU time
%     Out.iter:   Number of iterations
%     Out.obj :   Objective function value at Out.sol 
%--------------------------------------------------------------------------
% This code is programmed based on the algorithm proposed in 
% "S. Zhou, X. Xiu, Y. Wang, and D. Peng, 
%  Revisiting Lq(0<=q<1) norm regualrized optimization, 2023."
% Send your comments and suggestions to <<< slzhou2021@163.com >>> 
% Warning: Accuracy may not be guaranteed !!!!! 
%--------------------------------------------------------------------------

warning off;
t0  = tic;
if  nargin < 4
    disp(' No enough inputs. No problems will be solverd!'); return;
end
if nargin < 5; pars = [];  end 

[sig1,sig2,q1,q2,lamq,lamq1,alpha0,cg_tol,cg_it,x0,tol,...
          maxit,newton,show,prob] = setparameters(n,lambda, q, pars);
Fnorm    = @(var)norm(var,'fro')^2; 
Funcs    = @(x,T,key)funcF(x,T,key);
ProxLq   = @(x,t)ProxmLq(x,t,q);
Qnorm    = @(absvar)sum(absvar.^q); 
GQnorm   = @(var,absvar)(sign(var).*absvar.^q1); 
HQnorm   = @(absvar)(absvar.^q2); 
x        = x0; 
T        = find(x~=0); 
sT       = nnz(T);
w        = x(T);
if sT
    Obj  = Funcs(w,T,'f');
    grad = Funcs(w,T,'g'); 
else
    Obj  = Funcs(x,[],'f');
    grad = Funcs(x,[],'g'); 
end
 
sT       = nnz(T);
Error    = zeros(maxit,1); 
if  show 
    fprintf(' Start to run the solver -- PSNP \n');
    fprintf(' ----------------------------------------------------------------------------\n');
    fprintf(' Iter      ErrGrad        ErrProx        Objective      Sparsity     Time(sec) \n'); 
    fprintf(' ----------------------------------------------------------------------------\n');
    fprintf('%4d       %5.2e      %8.2e       %8.5e    %7d      %8.3f\n',...
                0, norm(grad,'inf'), 1, Obj, sT, toc(t0)); 
end

% The main body
for iter = 1:maxit
          
    switch prob
        case 'CS';  rate = 0.5; i0 = 6;
        case 'SVM'; rate = 0.1 + 0.1*(iter>5);
                    i0   = 5.0 + 5.0*(iter>5); 
        case 'LR';  rate = .25 + .25*(iter>5);
                    i0   =  15 + 10* (iter>5);       
        otherwise;  rate = 0.5; i0 = 25;
    end
   
    alpha       = alpha0; 
    Told        = T;
    for i       = 1:i0 
        [w,T]   = ProxLq(x-alpha*grad,alpha*lambda);        
        absw    = abs(w);  
        Objw    = Funcs(w,T,'f')+lambda*Qnorm(absw);  
        if isempty(T) || Objw < Obj - sig1*Fnorm(w-x(T))
           break; 
        end 
        alpha   = alpha*rate;
    end

    if isempty(T)
       alpha   = alpha/rate; 
       [w,T]   = ProxLq(x-alpha*grad,alpha*lambda); 
       absw    = abs(w);  
       Objw    = Funcs(w,T,'f')+lambda*Qnorm(absw);
    end
       
    x       = zeros(n,1);
    x(T)    = w;
    Objold  = Obj;
    Obj     = Objw; 
    sTold   = sT; 
    sT      = nnz(T); 
    ident   = 0;
    if sT   == sTold 
       ident = nnz(T-Told)==0;   
    end     
 
    if  newton && sT>0 &&  (sT <0.5*n || ident ==1) && (i~=i0 || iter>=5)
        [grad,Hess] = Funcs(w,T,'gh'); 
        gradT       = grad(T);
        if q        > 0
            gradT   = gradT + lamq*GQnorm(w,absw);
            dw      = lamq1*HQnorm(absw);
            if isa(Hess, 'function_handle') 
                Hess = @(v)(Hess(v)+dw.*v);
            else
                Hess(1:(sT+1):end) = Hess(1:(sT+1):end) + dw';
            end
        end
        if  isa(Hess, 'function_handle')    
            %cg_tol = 1e-16*sT; 
            if mod(iter,10)==0  
                cg_tol = max(cg_tol/10,1e-15*sT);
                cg_it  = min(cg_it+5,25);
            end    
            d   = my_cg(Hess,gradT,cg_tol,cg_it,zeros(sT,1));  
        else 
            d    = Hess\gradT; 
        end
        
        beta   = 1;
        Fd      = Fnorm(d);   
        for j   = 1 : 10
            v       = w - beta* d;
            absv    = abs(v);
            Objv    = Funcs(v,T,'f') + lambda*Qnorm(absv);          
            if  Objv <= Obj - sig2*beta^2*Fd 
                x(T) = v; 
                w    = v; 
                absw = absv;
                Obj  = Objv; 
                break;
            end 
            beta = beta * 0.25;
        end        
    end
    
    grad        = Funcs(w,T,'g'); 
    gradT       = grad(T);
    if q        > 0
        gradT   = gradT + lamq*GQnorm(w,absw);  
    end
    if isempty(T) || (sT==1 && isequal(prob,'SVM'))
        ErrGradT = 1e10; 
        lambda   = lambda/1.5; 
        lamq     = lambda*q;
        lamq1    = lamq*q1;
        if  isequal(prob,'SVM')
            x    = zeros(n,1);
            Obj  = Funcs(w,[],'f');
            grad = Funcs(w,[],'g');
        end
    else
        ErrGradT = norm(gradT,'inf'); 
    end
    ErrObj      = abs(Obj-Objold)/(1+abs(Obj));    
    Error(iter) = ErrGradT ;
    if  show  && mod(iter,10+90*(iter>100))==0
        fprintf('%4d       %5.2e      %8.2e       %8.5e    %7d      %8.3f\n',...
                iter,   ErrGradT, ErrObj, Obj, sT, toc(t0)); 
    end
             
    % Stopping criteria
     if  ident && max(ErrGradT,ErrObj) <tol
         if  show  && mod(iter,10+90*(iter>100))~=0
                fprintf('%4d       %5.2e      %8.2e       %8.5e    %7d      %8.3f\n',...
                iter,   ErrGradT, ErrObj, Obj, sT, toc(t0)); 
         end
         break;  
     end
end

fprintf(' ----------------------------------------------------------------------------\n');
Out.time    = toc(t0);
Out.iter    = iter;
Out.sol     = x;
Out.obj     = Obj;  
Out.Error   = Error; 
end

function [sig1,sig2,q1,q2,lamq,lamq1,alpha0,cg_tol,cg_it,x0,tol,...
          maxit,newton,show,prob] = setparameters(n,lambda, q, pars)

if isfield(pars,'x0');     x0     = pars.x0;     else; x0 = zeros(n,1); end 
if isfield(pars,'tol');    tol    = pars.tol;    else; tol    = 1e-6;   end  
if isfield(pars,'maxit');  maxit  = pars.maxit;  else; maxit  = 1e4;    end
if isfield(pars,'newton'); newton = pars.newton; else; newton = 1;      end 
if isfield(pars,'show');   show   = pars.show;   else; show   = 1;      end 
if isfield(pars,'prob');   prob   = pars.prob;   else; prob   = 'None'; end 

sig1  = 1e-6; 
sig2  = 1e-10;
q1    = q-1;
q2    = q-2;
lamq  = lambda*q;
lamq1 = lambda*q*(q-1);

switch prob
    case 'CS';   alpha0 = 1-q/2;
                 cg_tol = 1e-8; 
                 cg_it  = 10;
    case 'SVM';  alpha0 = 20;
                 cg_tol = 1e-6; 
                 cg_it  = 5;
   case 'LR';    alpha0 = max(1e4,10*sqrt(n));
                 cg_tol = 1e-6; 
                 cg_it  = 5;             
    otherwise;   alpha0 = 1;
                 cg_tol = 1e-8; 
                 cg_it  = 10;
end
end
% proximal operator of Lq norm ---------------------------------------------
function [px,T] = ProxmLq(a,alam,q)

switch q
    case 0
         t     = sqrt(2*alam);
         T     = find(abs(a)>t);  
         px    = a(T);
         
    case 1/2 
         t     = (3/2)*alam^(2/3);
         T     = find(abs(a) > t);
         aT    = a(T);
         phi   = acos( (alam/4)*(3./abs(aT)).^(3/2) );
         px    = (4/3)*aT.*( cos( (pi-phi)/3) ).^2;
         % px    = (2/3)*aT.*(1+ cos( 2*pi/3-(2/3)*phi ) );
        
    case 2/3
         t     = (128/27)^(1/4)*(alam)^(3/4);
         T     = find( abs(a) >  t );  
         aT    = a(T);       
         tmp1  = aT.^2/2; 
         tmp2  = sqrt( tmp1.^2 - (8*alam/9)^3 );  
         phi   = (tmp1+tmp2).^(1/3)+(tmp1-tmp2).^(1/3);
         px    = sign(aT)/8.*( sqrt(phi)+sqrt(2*abs(aT)./sqrt(phi)-phi) ).^3; 
      %  px  =  real(px);
    otherwise
        fprintf(' Input ''q'' is incorrect!!! \n');
        fprintf(' Please re-enter ''q'' one value of {0,1/2,2/3}!!!\n')
end

end


% conjugate gradient-------------------------------------------------------
function x = my_cg(fx,b,cgtol,cgit,x)
    r = b;
    e = norm(r,'fro')^2;
    t = e;
    for i = 1:cgit  
        if e < cgtol*t; break; end
        if  i == 1  
            p = r;
        else
            p = r + (e/e0)*p;
        end  
        if  isa(fx,'function_handle')
            w  = fx(p);
        else
            w  = fx*p;
        end
        a  = e/sum(p.*w);
        x  = x + a * p; 
        r  = r - a * w;  
        e0 = e;
        e  = norm(r,'fro')^2;
    end 
  
end
