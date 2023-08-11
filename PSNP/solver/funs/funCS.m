function [out1,out2] = funCS(xT,T,key, data)

    mark = isa(data.A, 'function_handle'); 
    if mark
        if ~isfield(data,'At') 
            disp('The transpose-data.At-is missing'); return; 
        end
        if ~isfield(data,'n')  
            disp('The dimension-data.n-is missing');  return;  
        end
        Atb = @(v)data.At(v);
    else
        Atb = @(v)(v'*data.A)';
    end

    if  isempty(T) 
        Axb = -data.b; 
        if isequal(key, 'f')
            out1 = norm(Axb,'fro')^2/2;
        else
            out1 = Atb(Axb);    
            if isequal(key,'gh'); out2 = []; end
        end            
    else    
        if ~mark         
            AT  = data.A(:,T);
            Axb = AT*xT-data.b; 
        else
            Axb = data.A(supp(data.n,xT,T))-data.b; 
        end

        if isequal(key, 'f')
            out1 = norm(Axb,'fro')^2/2;
        else
            out1 = Atb(Axb);
            if  isequal(key,'gh')  
                if ~mark
                    if nnz(T) < 500 && size(data.A,1)<500
                       out2 = AT'*AT;  
                    else
                       out2 = @(v)((AT*v)'*AT)' ;      
                    end 
                else
                    out2 = @(v)sub( Atb( data.A(supp(data.n,v,T))),T);
                end
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
