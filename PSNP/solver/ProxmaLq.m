function [px,T] = ProxmaLq(a,lam,q)

% solving problem   xopt = argmin_x 0.5*||x-a||^2 + lam*||x||_q^q

% a:    a vector
% lam:  a positive scalar
% q:    a scalar in [0,1)

% px:   px = xopt(T);
% T :   the support set of xopt

switch q
    case 0
         t     = sqrt(2*lam);
         T     = find(abs(a)>t);  
         px    = a(T);        
    case 1/2 
         t     = (3/2)*lam^(2/3);
         T     = find(abs(a) > t);
         aT    = a(T);
         phi   = acos( (lam/4)*(3./abs(aT)).^(3/2) );
         px    = (4/3)*aT.*( cos( (pi-phi)/3) ).^2;
    case 2/3
         t     = 2*(2*lam/3)^(3/4); 
         T     = find( abs(a) >  t );  
         aT    = a(T);       
         tmp1  = aT.^2/2; 
         tmp2  = sqrt( tmp1.^2 - (8*lam/9)^3 );  
         phi   = (tmp1+tmp2).^(1/3)+(tmp1-tmp2).^(1/3);
         px    = sign(aT)/8.*( sqrt(phi)+sqrt(2*abs(aT)./sqrt(phi)-phi) ).^3; 
    otherwise
         [px,T] = NewtonLq(a,lam,q);
end

end

function [w,T] = NewtonLq(a,alam,q)

    thresh = (2-q)*alam^(1/(2-q))*(2*(1-q))^((1-q)/(q-2));
    T      = find(abs(a)>thresh); 

    if ~isempty(T)
        zT     = a(T);
        w      = zT;
        maxit  = 1e2;
        q1     = q-1;
        q2     = q-2;
        lamq   = alam*q;
        lamq1  = lamq*q1;

        gradLq = @(u,v)(u - zT + lamq*sign(u).*v.^q1);
        hessLq = @(v)(1+lamq1*v.^q2);
        func   = @(u,v)(norm(u-zT)^2/2+alam*sum(v.^q));

        absw   = abs(w);
        fx0    = func(w,absw); 

        for iter  = 1:maxit
            g     = gradLq(w,absw);
            d     = -g./hessLq(absw); 
            alpha = 1;  
            w0    = w;
            for i    = 1:10
                w    =  w0 + alpha*d;  
                absw = abs(w);
                fx   = func(w,absw);
                if  fx < fx0 - 1e-4*norm(w-w0)^2
                   break; 
                end 
                alpha   = alpha*0.5;
            end
            if  norm(g) < 1e-8; break; end
        end
    else
        w = [];
    end

end

 