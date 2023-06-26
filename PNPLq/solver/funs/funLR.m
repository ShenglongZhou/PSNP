function [out1,out2] = 	funLR(xT,T,key,mu, data)

m     = length(data.b);    
if  isempty(T) 
    switch key
        case 'f'
             out1 = log(2);   
        case 'g'
             out1 = ((1/2-data.b)'*data.A)'/m; 
        case 'gh'
             out1 = ((1/2-data.b)'*data.A)'/m; 
             out2 = [];
    end
else
    AT  = data.A(:,T);
    Ax  = AT*xT; 
    eAx = exp(Ax);  
    switch key
        case 'f'
            if sum(eAx)==Inf 
                obj  = sum(log(1+eAx(Ax<=300)))+sum(Ax(Ax>300))-sum(data.b.*Ax);  
            else
                obj  = sum(log(1+eAx)-data.b.*Ax);    % objective function 
            end   
            out1    = obj/m + mu*norm(xT,'fro')^2/2;
        case 'g'
            dAx     = 1./(1+eAx);
            out1    = ((1-data.b-dAx)'*data.A)'/m;     % gradient   
            out1(T) =  out1(T) + mu*xT;
        case 'gh'
            dAx     = 1./(1+eAx);
            out1    = ((1-data.b-dAx)'*data.A)'/m;     % gradient
            out1(T) =  out1(T) + mu*xT;          
            d       = dAx.*(1-dAx)/m;          
            out2    =  @(v) ( ((d.*(AT*v))'*AT)' + mu*v) ;  % Hessian                  
    end            
end
    
end

 


