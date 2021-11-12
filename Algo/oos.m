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