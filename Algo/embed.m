function [V_SDP,V_DM,sqrt_eigenvalues_SDP,eigenvalues_DM,deg,K] = embed(X,id_train,bw,r,n_it,tol,nb_comp,dx_optional)

    % dx is the discretization step (only if we discretize an integral)
    if nargin > 7
        dx = dx_optional;
    else
        dx = 1;
    end
    
    d_tot = pdist2(X,X);
    N = size(X,1);

    n_train = length(id_train);

    % defining kernels
    X_train = X(id_train,:);
    d_train = pdist2(X_train,X_train);

    k = exp(-d_train.^2/bw^2);

    deg = dx*sum(k,2);
    v0 = sqrt(deg/sum(dx*deg));

    k_norm = diag(1./sqrt(deg))*k*diag(1./sqrt(deg));
    K = (k_norm-v0*v0');% diffusion kernel matrix

    % SDP solution
    diagonal = diag(K);

    % Initialization
    tic
    H0 = rand(n_train,r)-0.5; 
    H0 = diag(sqrt(diagonal))*H0./sqrt(sum(H0.^2,2));

    % Iterations
    [H,~] = ProjPowerIterations(K,H0,diagonal,n_it,tol);
    %[Answer,normgradient] = IsLocalMaximum(H,Q,diagonal); % checking optimality 

    % SVD 
    [V,L] = svd(H);
    l = diag(L); % singular value of H (sqrt eig of HH^T)
    nb_l = nnz(l>1e-08);

    sqrt_eigenvalues_SDP = l;

    fprintf('Eigenvalues of PCA H: %d \n',nb_l)
    disp(l(1:nb_l))
    toc

    % Plotting
    u0 =  l(1)*V(:,1);
    u1 =  l(2)*V(:,2);
    u2 =  l(3)*V(:,3);
    if nb_comp == 2
        V_SDP = [u0 u1];
    else
        V_SDP = [u0 u1 u2];
    end


    % Diffusion maps
    [X_c,Y_c] = eigs(K*dx);
    lambda_DM = diag(Y_c);

    eigenvalues_DM = lambda_DM;
    disp('eigenvalues of diffusion op')
    disp(lambda_DM)


    if nb_comp == 2
        if isnan(lambda_DM)
            % if spectral algo does not converge
            V_DM = ones(size(X_c(:,1:2)));
        else
            V_DM = X_c(:,1:2)*diag(lambda_DM(1:2));
        end
    else
        if isnan(lambda_DM)
            % if spectral algo does not converge
            V_DM = ones(size(X_c(:,1:3)));
        else
            V_DM = X_c(:,1:3)*diag(lambda_DM(1:3));
        end
    end

end

