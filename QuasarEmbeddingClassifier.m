clear; close all;
addpath('Data')
addpath('Algo')
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% Embedding of the interval [-1,1]
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

load HTRU_2.mat

X = zscore(X);
N = size(X,1);

bw = 10 % or 5
nb_nb = 5

d_tot = pdist2(X,X);

% selecting a subset of the digits to compute the embedding
n_train = floor(0.7*N);

n_rep = 3;
precision_plus_class_DM = zeros(n_rep,1);
recall_plus_class_DM = zeros(n_rep,1);

precision_plus_class_SDP = zeros(n_rep,1);
recall_plus_class_SDP = zeros(n_rep,1);

for rep = 1:n_rep

    id_train =  datasample(1:N,n_train,'Replace',false);
    y_train = Y(id_train);

    id_oos = setdiff(1:N,id_train);
    n_oos = length(id_oos);
    y_oos = Y(id_oos);


    n_it = 5000; % maximal number of iterations
    tol = 1e-09; % tolerance on relative difference between 2 iterates
    r = 30;
    nb_comp = 2;
    
    [V_SDP,V_DM,sqrt_eigenvalues_SDP,eigenvalues_DM,deg] = embed(X,id_train,bw,r,n_it,tol,nb_comp);

    
    s=3; % marker size 
    figure;scatter(V_SDP(:,1),V_SDP(:,2),s, y_train,'o','filled'); 
    colormap jet

    saveas(gcf,'Figures/quasarSDP','epsc')
    %

    figure; 
    scatter(V_DM(:,1),V_DM(:,2),s,y_train,'o','filled')
    colormap jet
    saveas(gcf,'Figures/quasarDiffusion','epsc')

    u_train = V_SDP;

    % Out-of-sample SDP embedding
    id_oos = setdiff(1:N,id_train);

    X_train = X(id_train,:);
    X_oos = X(id_oos,:);

    v0 = sqrt(deg/sum(deg));
    u_oos = oos(X_oos,X_train,u_train,deg,v0,bw);

    % extension  of diffusion embedding
    d_oos_x = pdist2(X_oos,X_train);
    k_oos = exp(-d_oos_x.^2/bw^2); 
    deg_oos = sum(k_oos,2);

    k_ext_norm = diag(1./sqrt(deg_oos))*k_oos*diag(1./sqrt(deg));
    lambda_DM = eigenvalues_DM;
    X_c = V_DM
    % Nyström-type extension
    X_c_oos_0 = (1/lambda_DM(1))*k_ext_norm*X_c(:,1);
    X_c_oos_1 = (1/lambda_DM(2))*k_ext_norm*X_c(:,2);

    X_c_oos = [X_c_oos_0 X_c_oos_1];
    X_c_train = [X_c(:,1) X_c(:,2)];

    % Classifier
    Mdl_DM = fitcknn(X_c_train,y_train,'NumNeighbors',nb_nb,'Standardize',0)

    pred_DM_oos = predict(Mdl_DM, X_c_oos);
    
    classes = [-1 , 1];
    [confus,numcorrect,precision_DM,recall_DM,F] = getcm(y_oos,pred_DM_oos,classes);
    precision_plus_class_DM(rep) = precision_DM(2);
    recall_plus_class_DM(rep) = recall_DM(2);

    scatter(X_c_oos_0,X_c_oos_1,s, y_oos,'o','filled'); 
    colormap jet

    saveas(gcf,'Figures/quasarDMoos','epsc')

   
    scatter(u_oos(:,1),u_oos(:,2),s, y_oos,'o','filled');    
    colormap jet

    saveas(gcf,'Figures/quasarSDPoos','epsc')

    % Classifier
    Mdl_SDP = fitcknn(u_train,y_train,'NumNeighbors',nb_nb,'Standardize',0)

    pred_SDP_oos = predict(Mdl_SDP, u_oos);
    
    classes = [-1 , 1];
    [confus,numcorrect,precision_SDP,recall_SDP,F] = getcm(y_oos,pred_SDP_oos,classes);
    precision_plus_class_SDP(rep) = precision_SDP(2);
    recall_plus_class_SDP(rep) = recall_SDP(2);

end


disp('precision plus class SDP')
mean(precision_plus_class_SDP)
std(precision_plus_class_SDP)

disp('recall plus class SDP')
mean(recall_plus_class_SDP)
std(recall_plus_class_SDP)


disp('precision plus class DM')
mean(precision_plus_class_DM)
std(precision_plus_class_DM)

disp('recall plus class DM')
mean(recall_plus_class_DM)
std(recall_plus_class_DM)