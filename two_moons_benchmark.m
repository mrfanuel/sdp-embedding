clear; close all;
addpath('Data')
addpath('Algo')
addpath('Utils')

% number of clusters
k_clusters = 2;
n_rep = 20; % number of times k-means is repeated

gaussians = false;
moons = true;

homemade = false

n_noise = 30; 
sigma = .5;
if homemade

    %% data generation
    center_up = [-1.; 0];
    center_down = [1.; 0];
    nb_sample = 500;
    radius = 1.5;
    noise_eps = 0.9;
    [X,Y] = gen_two_moons(center_up,center_down,radius,nb_sample,noise_eps);
    %
    center_up_noise = center_up' + [0 1.5];
    center_down_noise = center_down' + [0 -1.5];

    
    X_gaussian_up = center_up_noise + sigma*randn(n_noise,2);
    X_gaussian_down = center_down_noise  + sigma*randn(n_noise,2);

    X_gaussian = [X_gaussian_up;X_gaussian_down];
    Y_gaussian = [ones(n_noise,1);-ones(n_noise,1)];

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
else
    n = 500;
    %sig = 1/6;
    sig = 1/4;
    [X,Y] = twomoons_matlab(n,sig);

    X_gaussian_up = [-0.5 1.5] + sigma*randn(n_noise,2);
    X_gaussian_down = [0.5 -2]  + sigma*randn(n_noise,2);

        X_gaussian = [X_gaussian_down;X_gaussian_up];
        Y_gaussian = [ones(n_noise,1);2*ones(n_noise,1)];
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


end

X = zscore(X);

%% Plot raw data

figure;
scatter(X(:,1),X(:,2),[],Y,'.')
colormap jet

saveas(gcf,'Figures/twomoons','epsc')

%% Initialization

range_bw = 0.01:0.05:1;

n_spec = 4;
sqrt_eigs_SDP = zeros(length(range_bw),n_spec);
eigs_DM = zeros(length(range_bw),n_spec);


max_nmi_SDP = zeros(size(range_bw));
max_nmi_DM = zeros(size(range_bw));
max_nmi_DM_proj = zeros(size(range_bw));


mean_nmi_SDP = zeros(size(range_bw));
mean_nmi_DM = zeros(size(range_bw));
mean_nmi_DM_proj = zeros(size(range_bw));

median_nmi_SDP = zeros(size(range_bw));
median_nmi_DM = zeros(size(range_bw));
median_nmi_DM_proj = zeros(size(range_bw));

std_nmi_SDP = zeros(size(range_bw));
std_nmi_DM = zeros(size(range_bw));
std_nmi_DM_proj = zeros(size(range_bw));


for i = 1:length(range_bw)
    bw = range_bw(i);

    r = 30; % maimal rank of the solution
    n_it = 5000; % maximal number of iterations
    tol = 1e-09; % tolerance on relative difference between 2 iterates
    N = size(X,1);
    id_train = 1:N;
    [V_SDP,V_DM,sqrt_eigenvalues_SDP,eigenvalues_DM,~] = embed(X,id_train,bw,r,n_it,tol);

    sqrt_eigs_SDP(i,:) = sqrt_eigenvalues_SDP(1:n_spec);
    eigs_DM(i,:) = eigenvalues_DM(1:n_spec);


    V_DM_proj = V_DM;% normalize the rows of V_DM_proj
    V_DM_proj = V_DM_proj./sqrt(sum(abs(V_DM_proj).^2,2));

    V_SDP_proj = V_SDP;% normalize the rows of V_DM_proj
    V_SDP_proj = V_SDP_proj./sqrt(sum(abs(V_SDP_proj).^2,2));

    temp_nmi_SDP = zeros(n_rep,1);
    temp_nmi_SDP_proj = zeros(n_rep,1);

    temp_nmi_DM = zeros(n_rep,1);
    temp_nmi_DM_proj = zeros(n_rep,1);


    for j = 1: n_rep
        idx_SDP = kmeans(V_SDP,k_clusters);
        temp_nmi_SDP(j) = nmi(idx_SDP,Y);

        idx_SDP_proj = kmeans(V_SDP_proj,k_clusters);
        temp_nmi_SDP_proj(j) = nmi(idx_SDP_proj,Y);

        idx_DM = kmeans(V_DM,k_clusters);
        temp_nmi_DM(j) = nmi(idx_DM,Y);

        idx_DM_proj = kmeans(V_DM_proj,k_clusters);
        temp_nmi_DM_proj(j) = nmi(idx_DM_proj,Y);
    end

    max_nmi_SDP(i) = max(temp_nmi_SDP);
    max_nmi_SDP_proj(i) = max(temp_nmi_SDP_proj);

    max_nmi_DM(i) = max(temp_nmi_DM);
    max_nmi_DM_proj(i) = max(temp_nmi_DM_proj);


    mean_nmi_SDP(i) = mean(temp_nmi_SDP);
    mean_nmi_SDP_proj(i) = mean(temp_nmi_SDP_proj);

    mean_nmi_DM(i) = mean(temp_nmi_DM);
    mean_nmi_DM_proj(i) = mean(temp_nmi_DM_proj);

    median_nmi_SDP(i) = median(temp_nmi_SDP);
    median_nmi_SDP_proj(i) = median(temp_nmi_SDP_proj);

    median_nmi_DM(i) = median(temp_nmi_DM);
    median_nmi_DM_proj(i) = median(temp_nmi_DM_proj);

    std_nmi_SDP(i) = std(temp_nmi_SDP);
    std_nmi_SDP_proj(i) = std(temp_nmi_SDP_proj);

    std_nmi_DM(i) = std(temp_nmi_DM);
    std_nmi_DM_proj(i) = std(temp_nmi_DM_proj);

end


temp_nmi_input = zeros(n_rep,1);
for j = 1: n_rep
    idx_SDP = kmeans(X,k_clusters);
    temp_nmi_input(j) = nmi(idx_SDP,Y);

end
mean_nmi_input = mean(temp_nmi_input);
std_nmi_input = std(temp_nmi_input);

disp("--------------------")

figure;
errorbar(range_bw,mean_nmi_input*ones(size(range_bw)),std_nmi_input*ones(size(range_bw)),'k','DisplayName','kmeans')
ylim([0 1])
hold on;
errorbar(range_bw,mean_nmi_SDP,std_nmi_SDP,'-bo','DisplayName','mean SDP')
ylim([0 1])
errorbar(range_bw,mean_nmi_DM,std_nmi_DM,'-rs','DisplayName','mean DM')
ylim([0 1])
legend

saveas(gcf,'Figures/mean_twomoons_gaussians_benchmark','epsc')

figure;
errorbar(range_bw,mean_nmi_input*ones(size(range_bw)),std_nmi_input*ones(size(range_bw)),'k','DisplayName','kmeans')
ylim([0 1])
hold on;
errorbar(range_bw,mean_nmi_SDP_proj,std_nmi_SDP_proj,'-b*','DisplayName','mean SDP + proj')
ylim([0 1])
errorbar(range_bw,mean_nmi_DM_proj,std_nmi_DM_proj,'-rd','DisplayName','mean DM + proj')
ylim([0 1])
legend

saveas(gcf,'Figures/mean_proj_twomoons_gaussians_benchmark','epsc')

figure;
errorbar(range_bw,mean_nmi_input*ones(size(range_bw)),std_nmi_input*ones(size(range_bw)),'k','DisplayName','kmeans')
ylim([0 1])
hold on;
plot(range_bw,median_nmi_SDP,'-bo','DisplayName','median SDP')
ylim([0 1])
plot(range_bw,median_nmi_DM,'-rs','DisplayName','median DM')
ylim([0 1])
legend

saveas(gcf,'Figures/median_twomoons_gaussians_benchmark','epsc')

figure;
errorbar(range_bw,mean_nmi_input*ones(size(range_bw)),std_nmi_input*ones(size(range_bw)),'k','DisplayName','kmeans')
ylim([0 1])
hold on;
plot(range_bw,median_nmi_SDP_proj,'-bo','DisplayName','median SDP + proj')
ylim([0 1])
plot(range_bw,median_nmi_DM_proj,'-rd','DisplayName','median DM + proj')
ylim([0 1])
legend

saveas(gcf,'Figures/median_proj_twomoons_gaussians_benchmark','epsc')

figure;
plot(range_bw, sqrt_eigs_SDP)
saveas(gcf,'Figures/sqrt_eigs_SDP_twomoons_gaussians_benchmark','epsc')

figure;

plot(range_bw, eigs_DM)
saveas(gcf,'Figures/eigs_DM_twomoons_gaussians_benchmark','epsc')
