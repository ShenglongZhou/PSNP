function [out1,out2] = 	funSVM(xT,T,key,mu, data)
 
    m = length(data.b);
    if isfield(data,'At') 
        Atb   = @(v,ind)data.At(:,ind)*v;
    else 
        Atb   = @(v,ind)(v'*data.A(ind,:))';
    end
    
    if  isempty(T) 
        if isequal(key, 'f')
            out1 = 1/2;   
        else
            out1 = -Atb(data.b,1:m)/m; 
            if  isequal(key, 'gh')
                out2 = [];
            end
        end
    else 
        AT   = data.A(:,T);
        bAx  = 1-data.b.*(AT*xT);
        Tp   = (bAx>0); 
        bAxT = bAx(Tp); 
        if isequal(key, 'f')
            out1 = mu*norm( xT,'fro')^2/2 + norm(bAxT,'fro')^2/2/m;     % objective
        else
            out1    = - Atb(bAxT.*data.b(Tp),Tp)/m;                     
            out1(T) =  out1(T) +  mu*xT;                                % gradient           
            if  isequal(key, 'gh')
                s       = nnz(T); 
                ATpT    = AT(Tp,:);         
                if s    < 100 && nnz(Tp) < 100
                   out2 =  mu*speye(s) + (ATpT'*ATpT)/m;                % Hessian
                else
                   out2 =  @(v) (  mu*v + ((ATpT*v)'*ATpT)'/m ) ;  
                end   
            end                
        end            
    end
end
