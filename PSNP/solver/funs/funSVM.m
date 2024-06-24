function [out1,out2] = funSVM(xT,T,key,mu, data)
    n    = size(data.A,2);
    d    = ones(n,1); 
    d(n) = 0.01;
    if  isempty(T) 
        switch key
            case 'f';  out1 = 1/2; 
            case 'g';  out1 = -mu*(data.b'*data.A)'; 
            case 'gh'; out1 = -mu*(data.b'*data.A)'; 
                       out2 = [];
        end
    else
        AT   = data.A(:,T);
        dT   = d(T);
        dT2  = dT.*dT; 
        bAx  = 1-data.b.*(AT*xT);
        Tp   = find(bAx>0);
        bAxT = bAx(Tp); 
        switch key
            case 'f';  out1    = norm(dT.*xT)^2/2 + (mu/2)*norm(bAxT,'fro')^2;
            case 'g';  out1    = -mu*( (bAxT.*data.b(Tp))'*data.A(Tp,:) )';% gradient 
                       out1(T) = out1(T) + dT2.*xT;
            case 'gh'; out1    = -mu*( (bAxT.*data.b(Tp))'*data.A(Tp,:) )';% gradient 
                       out1(T) = out1(T) + dT2.*xT;
                       s       = nnz(T); 
                       ATpT    = AT(Tp,:); 
                       if   s  < 1000 && nnz(Tp)<1000 
                            out2 = mu*(ATpT'*ATpT);                        % Hessian
                            out2(1:s+1:end) = out2(1:s+1:end) + dT2'; 
                       else
                            out2 = @(v)( dT2.*v + mu*((ATpT*v)'*ATpT)' ); % Hessian
                       end 
        end 
    end
end
