function [out1,out2] = 	funLR(xT,T,key,mu, data)

    m  = length(data.b);  
    if isfield(data,'At') 
        Atb = @(v)data.At*v;
    else 
        Atb = @(v)(v'*data.A)';
    end
    
    if  isempty(T) 
        if isequal(key, 'f')
            out1 = log(2);  
        else
            out1 = Atb(1/2-data.b)/m;  
            if  isequal(key, 'gh')
                out2 = [];
            end
        end
    else 
        AT  = data.A(:,T);
        Ax  = AT*xT; 
        eAx = exp(Ax);  
        if isequal(key, 'f')
            if sum(eAx)==Inf 
                obj  = sum(log(1+eAx(Ax<=300)))+sum(Ax(Ax>300))-sum(data.b.*Ax);  
            else
                obj  = sum(log(1+eAx)-data.b.*Ax);     % objective function 
            end   
            out1    = obj/m + mu*norm(xT,'fro')^2/2;
        else
            dAx     = 1./(1+eAx);
            dAx1    = 1-dAx;
            out1    = Atb(dAx1-data.b)/m;                    
            out1(T) = out1(T) +  mu*xT;                        % gradient
            
            if  isequal(key, 'gh')
                d    = (dAx.*dAx1)/m;          
                out2 =  @(v) ( ((d.*(AT*v))'*AT)' + mu*v) ;       % Hessain
            end                
        end            
    end        
end

