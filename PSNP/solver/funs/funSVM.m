function [out1,out2] = 	funSVM(xT,T,key,mu, data)
 
    m = length(data.b);

    if  isempty(T) 
        switch key
            case 'f'
                 out1 = 1/2;   
            case 'g'
                 out1 = -(data.b'*data.A)'/m; 
            case 'gh'
                 out1 = -(data.b'*data.A)'/m; 
                 out2 = [];
        end
    else
     
        AT   = data.A(:,T);
        bAx  = 1-data.b.*(AT*xT);
        Tp   = (bAx>0);
        bAxT = bAx(Tp); 
        switch key
            case 'f'  
                out1    = mu*norm( xT,'fro')^2/2 + norm(bAxT,'fro')^2/2/m;
            case 'g'  
                out1    = -( (bAxT.*data.b(Tp))'*data.A(Tp,:) )'/m;     % gradient   
                out1(T) =  out1(T) +  mu*xT;
            case 'gh'
                out1    = -( (bAxT.*data.b(Tp))'*data.A(Tp,:) )'/m;     % gradient   
                out1(T) =  out1(T) +  mu*xT;

                s       = nnz(T); 
                ATpT    = AT(Tp,:);              
                if s    < .100 && nnz(Tp) < 100
                   out2 =  mu*speye(s) + (ATpT'*ATpT)/m;                % Hessian
                else
                   out2 =  @(v) (  mu*v + ((ATpT*v)'*ATpT)'/m ) ;  
                end   
                
        end            
    end
end

 


