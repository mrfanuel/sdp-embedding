clear; close all;
addpath('Data')
addpath('Algo')
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% Embedding of the interval [-1,1]
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

load HTRU_2.mat

X = zscore(X);

bw = 10
nb_nb = 5
%%
d_tot = pdist2(X,X);

N = size(X,1);


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

    % defining kernels
    X_train = X(id_train,:);
    d_train = pdist2(X_train,X_train);

    k = exp(-d_train.^2/bw^2);

    deg = sum(k,2);
    v0 = sqrt(deg/sum(deg));

    k_norm = diag(1./sqrt(deg))*k*diag(1./sqrt(deg));
    K = (k_norm-v0*v0');% diffusion kernel matrix


    % SDP solution
    diagonal = diag(K);

    n_it = 5000; % maximal number of iterations
    tol = 1e-09; % tolerance on relative difference between 2 iterates
    tic
    % Initialization
    r = 30;
    H0 = rand(n_train,r)-0.5; 
    H0 = diag(sqrt(diagonal))*H0./sqrt(sum(H0.^2,2));

    % Iterations
    [H,~] = ProjPowerIterations(K,H0,diagonal,n_it,tol);
    %[Answer,normgradient] = IsLocalMaximum(H,Q,diagonal); % checking optimality 

    % SVD 
    [V,L] = svd(H);
    l = diag(L);
    nb_l = nnz(l>1e-08);
    fprintf('Eigenvalues of PCA: %d \n',nb_l)
    disp(l(1:nb_l))
    toc
    % kernel matrix
    rho = H*H';

    % Plotting
    u0 =  l(1)*V(:,1);
    u1 =  l(2)*V(:,2);
    
    [X_c,Y_c] = eigs(k_norm);
    
    lambda_DM = diag(Y_c);

    
    s=3; % marker size %s = mean(deg)*deg/(sum(deg));
    figure;scatter(u0,u1,s, y_train,'o','filled'); %title('SDP embedding')       
    %colorbar;
    colormap jet

    saveas(gcf,'quasarSDP','epsc')
    %

    figure; 
    scatter(X_c(:,2),X_c(:,3),s,y_train,'o','filled')
    %colorbar; 
    colormap jet
    %title('Diffusion embedding')   
    saveas(gcf,'quasarDiffusion','epsc')


    %


    u_train = [u0 u1];

    dim = 2;
    id_oos = setdiff(1:N,id_train);

    % initialization
    %u_oos = zeros(n_oos,dim);
    %d = zeros(n_train,1);

    % Out-of-sample
    X_train = X(id_train,:);
    X_oos = X(id_oos,:);

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

    % extension  DM

    X_c_oos_0 = (1/lambda_DM(2))*k_ext_norm*X_c(:,2);
    X_c_oos_1 = (1/lambda_DM(3))*k_ext_norm*X_c(:,3);

    X_c_oos = [X_c_oos_0 X_c_oos_1];
    X_c_train = [X_c(:,2) X_c(:,3)];

    % Classifier
    Mdl_DM = fitcknn(X_c_train,y_train,'NumNeighbors',nb_nb,'Standardize',0)

    pred_DM_oos = predict(Mdl_DM, X_c_oos);
    
    classes = [-1 , 1];
    [confus,numcorrect,precision_DM,recall_DM,F] = getcm(y_oos,pred_DM_oos,classes);
    precision_plus_class_DM(rep) = precision_DM(2);
    recall_plus_class_DM(rep) = recall_DM(2);

    scatter(X_c_oos_0,X_c_oos_1,s, y_oos,'o','filled'); %title('SDP embedding')       
    %colorbar;
    colormap jet

    saveas(gcf,'quasarDMoos','epsc')

   
    scatter(u_oos(:,1),u_oos(:,2),s, y_oos,'o','filled'); %title('SDP embedding')       
    %colorbar;
    colormap jet

    saveas(gcf,'quasarSDPoos','epsc')

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

disp('precision plus class SDP')
mean(recall_plus_class_SDP)
std(recall_plus_class_SDP)


disp('precision plus class DM')
mean(precision_plus_class_DM)
std(precision_plus_class_DM)

disp('precision plus class DM')
mean(recall_plus_class_DM)
std(recall_plus_class_DM)