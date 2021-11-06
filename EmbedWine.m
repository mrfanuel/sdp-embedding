%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%Code embedding the wine dataset
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
clear;close all;
clc

addpath(genpath('Data'))  
addpath(genpath('Algo'))
addpath(genpath('Utils'))  

%%%%%%%%%%%%%%%%%%%%%%%%%% Loading training data of MNIST %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
[x,t] = wine_dataset;
n_train = size(t,2);
t = t';
x = x';
x = zscore(x); % standardization

id1 = (t(:,1)==1);
id2 = (t(:,2)==1);
id3 = (t(:,3)==1);
truth = ones(n_train ,1);
truth(id1)=1;
truth(id2)=2;
truth(id3)=3;

%%%%%%%%%%%%%%%%%%%%%%%%%% Computing Kernel matrices %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% squared Euclidean distance matrix
D2 = squareform(pdist(x)).^2;
 
range_bw = 0.7:0.1:10;
nb_pts = length(range_bw);
spectrum = zeros(nb_pts,3);
FroNormDifferenceIterate = zeros(nb_pts,1);


for i=1:nb_pts
    bw = range_bw(i);
    k = exp(-D2/bw^2);
    deg = sum(k,2);
    v0 = sqrt(deg/sum(deg));

    k_norm = diag(1./sqrt(deg))*k*diag(1./sqrt(deg));
    K = k_norm-v0*v0';
    diagonal = diag(K);

    %%%%%%%%%%%%%%%%%%%%%%%%% Projected Power Method %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    % Computing a solution of the low rank fact. of (SDP) via  Projected Power Method

    n_it = 1e6; % max number of iterations
    tol = 1e-06; % stopping criterion

    %% Initialization
    r0 = 20;    
    id =  datasample(1:n_train,r0,'Replace',false);
    H0 = k(:,id)-v0*(v0(id))';
    H0 = sparse(1:n_train,1:n_train,sqrt(diagonal))*H0./sqrt(sum(H0.^2,2));

    %% Projected Power Method
    [H,dist,~] = ProjPowerIterations(K,H0,diagonal,n_it,tol);
    FroNormDifferenceIterate(i) = dist;
    [V,L] = svd(H);

    % disp('Singular values of the solution')
    l0 = diag(L);
    % disp(l0)
    disp('Nb of significant singular values')
    [id,~] = find(l0>1e-05);
    nb_nnz = nnz(l0>1e-05);
    if nnz(l0)>=3
        spectrum(i,:) = l0(1:3).^2/sum(l0.^2);
    elseif nnz(l0)==2
        spectrum(i,1:2) = l0(1:2).^2/sum(l0.^2);
    elseif nnz(l0)==1
        spectrum(i,1) = l0(1).^2/sum(l0.^2);
    end
    
    
    if bw == 1.5
        u0 =  l0(1)*V(:,1);
        u1 =  l0(2)*V(:,2);
        u_train = [u0 u1];
        figure;scatter(u0,u1,[],truth,'.'); title('SDP embedding sigma=1.5')
        colormap jet;
        place = '/Figures/Embedding_Wine_1p5.png';
        saveas(gcf,place);
        close all;
    end
    if bw == 10
        u0 =  l0(1)*V(:,1);
        u1 =  l0(2)*V(:,2);
        u_train = [u0 u1];
        figure;scatter(u0,u1,[],truth,'.'); title('SDP embedding sigma=10')
        colormap jet;
        place = '/Figures/Embedding_Wine_10.png';
        saveas(gcf,place);
        close all;
    end
    
end

%%
figure;
plot(range_bw,spectrum(:,1),'.');
hold on;
plot(range_bw,spectrum(:,2),'.');
plot(range_bw,spectrum(:,3),'.');
place = '/Figures/Spetrum_Wine.png';
saveas(gcf,place);
close all;

figure;
plot(range_bw,FroNormDifferenceIterate,'.');
title('Frobenius norm of difference between last two iterate')
place = '/Figures/Convergence_Wine.png';
saveas(gcf,place);
close all;


% %%
% %%%%%%%%%%%%%%%%%%%%%%% % Embedding of the training set %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% figure;
% u0 =  l0(1)*V(:,1);
% u1 =  l0(2)*V(:,2);
% u_train = [u0 u1];
% figure;scatter(u0,u1,[],truth,'.'); title('SDP embedding')
% colormap jet;
% %place = '/Figures/wine_SDP_embeding.png';
% %saveas(gcf,place)
% %close all;
