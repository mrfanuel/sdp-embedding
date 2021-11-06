function [Answer,normgradient] = IsLocalMaximum(H,Q,diagonal)
    p = H*H';
    S =  diag(1./diagonal)*diag(sum(p.*Q,2))-Q; 
    normgradient = norm(S*H);
    fprintf('Norm of Riemanian gradient %d \n',normgradient)
    [Lmin] = eigs(S,6,'SM');
    disp('Check the 6 eigenvalues of the Hessian of smallest magnitudes')
    disp(Lmin)
    if nnz(Lmin<-1e-6)==0
        Answer = 1;
        disp('Second order critical point')
    else
        Answer = 0;
        disp('It is a Saddle point')
    end
    
end