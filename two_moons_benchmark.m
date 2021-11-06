clear; close all;
addpath('Data')
addpath('Algo')
addpath('Utils')

% number of clusters
k_clusters = 2;
n_rep = 10; % number of times k-means is repeated


%% data generation
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

gaussians = false;
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

%% Plot raw data

figure;
scatter(X(:,1),X(:,2),[],Y,'.')
colormap jet

saveas(gcf,'/oos.mFigures/twomoons','epsc')

%% Initialization

range_bw = 0.05:0.1:2.05;

%percentage_2_leading_eig_SDP = zeros(size(range_bw));


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

    [V_SDP,V_DM,sqrt_eigenvalues_SDP,eigenvalues_DM] = embed(X,id_train,bw,r,n_it,tol)

    V_DM_proj = V_DM;% normalize the rows of V_DM_proj
    V_DM_proj = V_DM_proj./sqrt(sum(abs(V_DM_proj).^2,2));

    n_rep = 10;
    temp_nmi_SDP = zeros(n_rep,1);
    temp_nmi_DM = zeros(n_rep,1);
    temp_nmi_DM_proj = zeros(n_rep,1);


    for j = 1: n_rep
        idx_SDP = kmeans([u0 u1],k_clusters);
        temp_nmi_SDP(j) = nmi(idx_SDP,Y);

        idx_DM = kmeans(V_DM,k_clusters);
        temp_nmi_DM(j) = nmi(idx_DM,Y);

        idx_DM_proj = kmeans(V_DM_proj,k_clusters);
        temp_nmi_DM_proj(j) = nmi(idx_DM_proj,Y);
    end

    max_nmi_SDP(i) = max(temp_nmi_SDP);
    max_nmi_DM(i) = max(temp_nmi_DM);
    max_nmi_DM_proj(i) = max(temp_nmi_DM_proj);


    mean_nmi_SDP(i) = mean(temp_nmi_SDP);
    mean_nmi_DM(i) = mean(temp_nmi_DM);
    mean_nmi_DM_proj(i) = mean(temp_nmi_DM_proj);

    median_nmi_SDP(i) = median(temp_nmi_SDP);
    median_nmi_DM(i) = median(temp_nmi_DM);
    median_nmi_proj(i) = median(temp_nmi_DM_proj);

    std_nmi_SDP(i) = std(temp_nmi_SDP);
    std_nmi_DM(i) = std(temp_nmi_DM);
    std_nmi_DM_proj(i) = std(temp_nmi_DM_proj);

end

disp("[max_nmi_SDP ;max_nmi_DM; max_nmi_DM_proj]")
disp([max_nmi_SDP ;max_nmi_DM; max_nmi_DM_proj])


disp("[mean_nmi_SDP ;mean_nmi_DM]")
disp([mean_nmi_SDP ;mean_nmi_DM; mean_nmi_DM_proj])

disp("[std_nmi_SDP ;std_nmi_DM ;std_nmi_DM_proj]")
disp([std_nmi_SDP ;std_nmi_DM; std_nmi_DM_proj])

disp("[median_nmi_SDP ;median_nmi_DM; median_nmi_DM_proj]")
disp([median_nmi_SDP ;median_nmi_DM; median_nmi_DM_proj])



temp_nmi_input = zeros(n_rep,1);

for j = 1: n_rep
    idx_SDP = kmeans(X,k_clusters);
    temp_nmi_input(j) = nmi(idx_SDP,Y);

end

disp("max(temp_nmi_input)")
disp(max(temp_nmi_input))

disp("mean(temp_nmi_input)")
mean_nmi_input = mean(temp_nmi_input)
disp(mean_nmi_input)

disp("std(temp_nmi_input)")
std_nmi_input = std(temp_nmi_input)
disp(std_nmi_input)

disp("median(temp_nmi_input)")
disp(median(temp_nmi_input))

disp("--------------------")

figure;
errorbar(range_bw,mean_nmi_input*ones(size(range_bw)),std_nmi_input*ones(size(range_bw)),'k','DisplayName','kmeans')
ylim([0 1])
hold on;
errorbar(range_bw,mean_nmi_SDP,std_nmi_SDP,'-bo','DisplayName','mean SDP')
ylim([0 1])
errorbar(range_bw,mean_nmi_DM,std_nmi_DM,'-rs','DisplayName','mean DM')
ylim([0 1])
errorbar(range_bw,mean_nmi_DM_proj,std_nmi_DM_proj,'-rd','DisplayName','mean DM + proj')
ylim([0 1])
legend

saveas(gcf,'/Figures/mean_twomoons_gaussians_benchmark','epsc')

figure;
errorbar(range_bw,mean_nmi_input*ones(size(range_bw)),std_nmi_input*ones(size(range_bw)),'k','DisplayName','kmeans')
ylim([0 1])
hold on;
plot(range_bw,median_nmi_SDP,'-bo','DisplayName','median SDP')
ylim([0 1])
plot(range_bw,median_nmi_DM,'-rs','DisplayName','median DM')
ylim([0 1])
plot(range_bw,median_nmi_DM_proj,'-rd','DisplayName','median DM + proj')
ylim([0 1])
legend

saveas(gcf,'/Figures/median_twomoons_gaussians_benchmark','epsc')