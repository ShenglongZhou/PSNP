function [out1,out2] = funCS(xT,T,key, data)

mark = isa(data.A, 'function_handle'); 
if mark
    if ~isfield(data,'At') 
        disp('The transpose-data.At-is missing'); return; 
    end
    if ~isfield(data,'n')  
        disp('The dimension-data.n-is missing');  return;  
    end
end

if  isempty(T) 
    Axb = -data.b; 
    switch key 
        case 'f'
             out1 = norm(Axb,'fro')^2/2;    
        case 'g'
             if ~mark
                 out1 = (Axb'*data.A)';
             else
                 out1 = data.At(Axb);
             end
           
        case 'gh'
             if ~mark
                 out1 = (Axb'*data.A)';
             else
                 out1 = data.At(Axb);
             end
             out2 = [];
    end            
else  
    if ~mark         
        AT  = data.A(:,T);
        Axb = AT*xT-data.b;         
        switch key
            case 'f'
                out1 = norm(Axb,'fro')^2/2;    
            case 'g'
                out1 = (Axb'*data.A)';
            case 'gh'
                out1 = (Axb'*data.A)'; 
                if nnz(T) < 500 && size(data.A,1)<1e4
                   out2 = AT'*AT;  
                else
                   out2 = @(v)((AT*v)'*AT)' ;      
                end 
        end                    
    else  
        Axb  = data.A(supp(data.n,xT,T))-data.b;  
        switch key
            case 'f'
                out1 = norm(Axb,'fro')^2/2;    
            case 'g'
                out1 = data.At(Axb);
            case 'gh'
                out1 = data.At(Axb); 
                out2 = @(v)sub( data.At( data.A(supp(data.n,v,T))),T);      
        end
    end
end
end


function z = supp(n,x,T)
    z      = zeros(n,1);
    z(T)   = x;
end

function subz = sub(z,T)
         subz = z(T,:);
end



