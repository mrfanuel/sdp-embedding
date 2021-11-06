clear; close all;
addpath('Data')
addpath('Algo')
addpath('Utils')  

bw = 0.2;

% number of clusters
k_clusters = 2;
n_rep = 10; % number of times k-means is repeated


center_up = [-1.; 0];
center_down = [1.; 0];
nb_sample = 500;
radius = 1.5;
noise_eps = 0.9;

[X,Y] = gen_two_moons(center_up,center_down,radius,nb_sample,noise_eps);
%
n_noise = 50; 

center_up_noise = center_up' + [0 1.5];
center_down_noise = center_down' + [0 -1.5];

X_gaussian_up = center_up_noise + 1.5*randn(n_noise,2);
X_gaussian_down = center_down_noise  + 1.5*randn(n_noise,2);

X_gaussian = [X_gaussian_up;X_gaussian_down];
Y_gaussian = [ones(n_noise,1);-ones(n_noise,1)];

gaussians = true;
moons = true;


if gaussians && ~moons
    disp("only gaussians")
    X = X_gaussian;
    Y = Y_gaussian;    
elseif moons && gaussians
    disp("gaussians and moons")
    X = [X ;X_gaussian];
    Y = [Y; Y_gaussian];
elseif moons && ~gaussians
    disp("only moons")
end

X = zscore(X);


figure;
scatter(X(:,1),X(:,2),[],Y,'.')
colormap jet

saveas(gcf,'twomoons','epsc')

disp("bw")
disp(bw)
N = size(X,1);
id_train = 1:N;
y_train = Y(id_train);

r = 30; % maimal rank of the solution
n_it = 5000; % maximal number of iterations
tol = 1e-09; % tolerance on relative difference between 2 iterates

[V_SDP,V_DM,sqrt_eigenvalues_SDP,eigenvalues_DM] = embed(X,id_train,bw,r,n_it,tol)


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



