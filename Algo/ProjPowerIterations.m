function [H,dist,Obj] = ProjPowerIterations(Q,H0,diagonal,n_it,tol)
disp('-------------------------------------')
disp('Projected power method ...')
tic
H = H0;
dist = 1;
N = length(Q);

Q = Q+1e-07*eye(N);
d = sqrt(diagonal);

for l =1:n_it
    
    % power iteration
    H1 = Q*H; 
    H1 = sparse(1:N,1:N,d)*H1./sqrt(sum(abs(H1).^2,2)); % normalize the rows of H1    
    
    % build gradient
    dist = norm(H1-H,'fro');
    
    % stopping criterion
    if dist< tol
        break;
    end
    
    H = H1;
    if ~mod(l/1000,1)
        fprintf('iteration: %d \n', l)
    end
    Obj(l) = trace(H'*Q*H);
end
if l==n_it+1
    fprintf('Max number of iteration attained, tol achieved: %d \n', dist)
else
    fprintf('Distance between successive iterates %d achieved after %d iterations \n',dist, l)
end
toc
end