function u_oos = oos(X_oos,X,id_train,u_train,deg,v0,bw)

        n_train = length(id_train);
        for j=1:n_train
            d(j) = norm(X_oos-X(id_train(j),:),2);
        end
        k_oos = exp(-d.^2/bw^2)'; 
        deg_oos = sum(k_oos,2);
        k_oos_norm = (1/sqrt(deg_oos))*k_oos*diag(1./sqrt(deg));
        v0_oos = k_oos_norm*v0;

        K_oos = k_oos_norm -v0_oos*v0';
        nor = 1/deg_oos-deg_oos/(sum(deg));
        u_oos0 = K_oos*u_train;
        M = sum(u_oos0.^2);
        u_oos = sqrt(nor)*u_oos0/sqrt(M);
end