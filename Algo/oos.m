
%{
 function u_oos = oos(X_oos,X,id_train,u_train,deg,v0,bw)
    n_oos = size(X_oos,1);
    n_train = length(id_train);
    for i=1:n_oos
        x_oos = X_oos(i,:);
        for j=1:n_train
            d(j) = norm(x_oos-X(id_train(j),:),2);
        end
        k_oos = exp(-d.^2/bw^2)'; 
        deg_oos = sum(k_oos,2);
        k_oos_norm = (1/sqrt(deg_oos))*k_oos*diag(1./sqrt(deg));
        v0_oos = k_oos_norm*v0;

        K_oos = k_oos_norm -v0_oos*v0';
        nor = 1/deg_oos-deg_oos/(sum(deg));
        u_oos0 = K_oos*u_train;
        M = sum(u_oos0.^2);
        u_oos(i,:) = sqrt(nor)*u_oos0/sqrt(M);
    end    

end 
%}


function u_oos = oos(X_oos,X_train,u_train,deg,v0,bw)

    d_oos_x = pdist2(X_oos,X_train);
    k_oos = exp(-d_oos_x.^2/bw^2); 
    deg_oos = sum(k_oos,2);
    k_ext_norm = diag(1./sqrt(deg_oos))*k_oos*diag(1./sqrt(deg));
    v0_ext = k_ext_norm*v0;

    K_ext = k_ext_norm -v0_ext*v0';
    nor = 1./deg_oos-deg_oos/(sum(deg));
    u_oos_0 = K_ext*u_train;
    M = sum(u_oos_0.^2,2);
    u_oos = diag(sqrt(nor))*u_oos_0./sqrt(M);
end