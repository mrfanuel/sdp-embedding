%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%Code embedding the wine dataset
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
clear;close all;

addpath(genpath('Data'))  
addpath(genpath('Algo'))
addpath(genpath('Utils'))  

%%%%%%%%%%%%%%%%%%%%%%%%%% Loading training data of MNIST %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
[X,Y] = wine_dataset;
N = size(Y,2);
Y = Y';
X = X';
X = zscore(X); % standardization

id1 = (Y(:,1)==1);
id2 = (Y(:,2)==1);
id3 = (Y(:,3)==1);
truth = ones(N ,1);
truth(id1)=1;
truth(id2)=2;
truth(id3)=3;

 
range_bw = 0.7:0.1:10;
nb_bw = length(range_bw);
spectrum = zeros(nb_bw,3);


for i=1:nb_bw
    
    bw = range_bw(i);
    n_it = 5000; % max number of iterations
    tol = 1e-06; % stopping criterion

    %% Initialization
    r = 20;
    id_train = 1:N;
    n_comp = 3;
    [V_SDP,V_DM,sqrt_eigenvalues_SDP,eigenvalues_DM] = embed(X,id_train,bw,r,n_it,tol,n_comp);
    l0 = sqrt_eigenvalues_SDP;

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
        figure;scatter(V_SDP(:,1),V_SDP(:,2),[],truth,'.'); title('SDP embedding sigma=1.5')
        colormap jet;
        saveas(gcf,'Figures/Embedding_Wine_1p5','epsc')
        close all;
    end
    if bw == 10
        figure;scatter(V_SDP(:,1),V_SDP(:,2),[],truth,'.'); title('SDP embedding sigma=10')
        colormap jet;
        saveas(gcf,'Figures/Embedding_Wine_10','epsc')
        close all;
    end
    
end

%%
figure;
plot(range_bw,spectrum(:,1),'.');
hold on;
plot(range_bw,spectrum(:,2),'.');
plot(range_bw,spectrum(:,3),'.');
saveas(gcf,'Figures/Spetrum_Wine','epsc');
close all;




