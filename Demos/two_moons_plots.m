clear; close all;
addpath('../Data')
addpath('../Algo')
addpath('../Utils')  

bw = 0.1

% number of clusters
k_clusters = 2;
n_rep = 1; % number of times k-means is repeated


sigma = .2;
n = 200;

sig = 1/6
[X,Y] = twomoons_matlab(n,sig);
X = zscore(X);


figure;
scatter(X(:,1),X(:,2),[],Y,'.')
colormap jet
saveas(gcf,'Figures/twomoons','epsc')

disp("bw")
disp(bw)
N = size(X,1);
id_train = 1:N;
y_train = Y(id_train);

r = 30; % maimal rank of the solution
n_it = 5000; % maximal number of iterations
tol = 1e-09; % tolerance on relative difference between 2 iterates
n_comp = 3;
[V_SDP,V_DM,sqrt_eigenvalues_SDP,eigenvalues_DM,~,~] = embed(X,id_train,bw,r,n_it,tol,n_comp)


V_DM_proj = V_DM;% normalize the rows of V_DM_proj
V_DM_proj = V_DM_proj./sqrt(sum(abs(V_DM_proj).^2,2));

idx_SDP = kmeans(V_SDP,k_clusters);
idx_DM = kmeans(V_DM,k_clusters);
idx_DM_proj = kmeans(V_DM_proj,k_clusters);


k = 2
idx_SDP = kmeans(V_SDP,k);

disp('nmi SDP')
disp(nmi(idx_SDP,Y))

idx_DM = kmeans(V_DM,k);

disp('nmi DM')
disp(nmi(idx_DM,Y))

idx_DM_proj = kmeans(V_DM_proj,k);

disp('nmi DM proj')
disp(nmi(idx_DM_proj,Y))

s=3; 
id1 = find(idx_SDP==1);
id2 = find(idx_SDP==2);
figure;
scatter(V_SDP(id1,1),V_SDP(id1,2),s, y_train(id1),'o','filled'); 
hold on;
scatter(V_SDP(id2,1),V_SDP(id2,2),s, y_train(id2),'s'); 
colormap jet
saveas(gcf,'Figures/twomoonsSDP','epsc')

id1 = find(idx_DM==1);
id2 = find(idx_DM==2);
figure;
scatter(V_DM(id1,1),V_DM(id1,2),s, y_train(id1),'o','filled'); 
hold on;
scatter(V_DM(id2,1),V_DM(id2,2),s, y_train(id2),'s'); 
colormap jet
saveas(gcf,'Figures/twomoonsDiffusion','epsc')

id1 = find(idx_DM_proj==1);
id2 = find(idx_DM_proj==2);
figure;
scatter(V_DM_proj(id1,1),V_DM_proj(id1,2),s, y_train(id1),'o','filled'); 
hold on;
scatter(V_DM_proj(id2,1),V_DM_proj(id2,2),s, y_train(id2),'s'); 
colormap jet
saveas(gcf,'Figures/twomoonsDiffusionProj','epsc')



