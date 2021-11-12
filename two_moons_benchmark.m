clear; close all;
addpath('Data')
addpath('Algo')
addpath('Utils')

% number of clusters
k_clusters = 2;


n = 200; % 200 good for sparsity
nb_datasets = 10
n_spec = 4;

range_bw = 0.05:0.05:1;
nb_bw = length(range_bw);

sqrt_eigs_SDP = zeros(nb_bw,n_spec);
eigs_DM = zeros(nb_bw,n_spec);


max_nmi_SDP = zeros(nb_bw,1);
max_nmi_DM = zeros(nb_bw,1);
max_nmi_DM_proj = zeros(nb_bw,1);


mean_nmi_SDP = zeros(nb_bw,1);
mean_nmi_DM = zeros(nb_bw,1);
mean_nmi_DM_proj = zeros(nb_bw,1);

median_nmi_SDP = zeros(nb_bw,1);
median_nmi_DM = zeros(nb_bw,1);
median_nmi_DM_proj = zeros(nb_bw,1);

std_nmi_SDP = zeros(nb_bw,1);
std_nmi_DM = zeros(nb_bw,1);
std_nmi_DM_proj = zeros(nb_bw,1);



for m = 1:nb_datasets
    sig = 1/6;
    [X,Y] = twomoons_matlab(n,sig);
    X = zscore(X);

    %% Plot raw data
    figure;
    scatter(X(:,1),X(:,2),[],Y,'.')
    colormap jet
    saveas(gcf,'Figures/twomoons','epsc')

    idx_input = kmeans(X,k_clusters);
    temp_nmi_input(m) = nmi(idx_input,Y);

    for i = 1:length(range_bw)
        bw = range_bw(i);
        disp(bw)
        r = 30; % maximal rank of the solution
        n_it = 5000; % maximal number of iterations
        tol = 1e-09; % tolerance on relative difference between 2 iterates
        N = size(X,1);
        id_train = 1:N;
        nb_comp = 2;
        [V_SDP,V_DM,sqrt_eigenvalues_SDP,eigenvalues_DM,~,~] = embed(X,id_train,bw,r,n_it,tol,nb_comp);

        eigenvalues_SDP = sqrt_eigenvalues_SDP.^2
        eigs_SDP(i,:) = eigenvalues_SDP(1:n_spec)/sum(eigenvalues_SDP);
        eigs_DM(i,:) = eigenvalues_DM(1:n_spec);

        V_DM_proj = V_DM;% normalize the rows of V_DM_proj
        V_DM_proj = V_DM_proj./sqrt(sum(abs(V_DM_proj).^2,2));

        V_SDP_proj = V_SDP;% normalize the rows of V_DM_proj
        V_SDP_proj = V_SDP_proj./sqrt(sum(abs(V_SDP_proj).^2,2));

        idx_SDP = kmeans(V_SDP,k_clusters);
        temp_nmi_SDP(i,m) = nmi(idx_SDP,Y);

        idx_SDP_proj = kmeans(V_SDP_proj,k_clusters);
        temp_nmi_SDP_proj(i,m) = nmi(idx_SDP_proj,Y);

        idx_DM = kmeans(V_DM,k_clusters);
        temp_nmi_DM(i,m) = nmi(idx_DM,Y);

        idx_DM_proj = kmeans(V_DM_proj,k_clusters);
        temp_nmi_DM_proj(i,m) = nmi(idx_DM_proj,Y);

    end

end

disp("--------------------")

mean_nmi_input = mean(temp_nmi_input);
std_nmi_input = std(temp_nmi_input);

for i = 1:length(range_bw)
    max_nmi_SDP(i) = max(temp_nmi_SDP(i,:));
    max_nmi_SDP_proj(i) = max(temp_nmi_SDP_proj(i,:));

    max_nmi_DM(i) = max(temp_nmi_DM(i,:));
    max_nmi_DM_proj(i) = max(temp_nmi_DM_proj(i,:));


    mean_nmi_SDP(i) = mean(temp_nmi_SDP(i,:));
    mean_nmi_SDP_proj(i) = mean(temp_nmi_SDP_proj(i,:));

    mean_nmi_DM(i) = mean(temp_nmi_DM(i,:));
    mean_nmi_DM_proj(i) = mean(temp_nmi_DM_proj(i,:));

    median_nmi_SDP(i) = median(temp_nmi_SDP(i,:));
    median_nmi_SDP_proj(i) = median(temp_nmi_SDP_proj(i,:));

    median_nmi_DM(i) = median(temp_nmi_DM(i,:));
    median_nmi_DM_proj(i) = median(temp_nmi_DM_proj(i,:));

    std_nmi_SDP(i) = std(temp_nmi_SDP(i,:));
    std_nmi_SDP_proj(i) = std(temp_nmi_SDP_proj(i,:));

    std_nmi_DM(i) = std(temp_nmi_DM(i,:));
    std_nmi_DM_proj(i) = std(temp_nmi_DM_proj(i,:));
end

figure;
errorbar(range_bw,mean_nmi_input*ones(size(range_bw)),std_nmi_input*ones(size(range_bw)),'k','DisplayName','kmeans raw data')
ylim([0 1])
hold on;
errorbar(range_bw,mean_nmi_SDP,std_nmi_SDP,'-bo','DisplayName','mean SDP')
ylim([0 1])
errorbar(range_bw,mean_nmi_DM,std_nmi_DM,'-rs','DisplayName','mean DM')
ylim([0 1])
legend('nmi kmeans raw data','nmi SDP','nmi DM')

matlab2tikz('Figures/mean_twomoons_gaussians_benchmark.tikz')
saveas(gcf,'Figures/mean_twomoons_gaussians_benchmark','epsc')

figure;
errorbar(range_bw,mean_nmi_input*ones(size(range_bw)),std_nmi_input*ones(size(range_bw)),'k','DisplayName','kmeans raw data')
ylim([0 1])
hold on;
errorbar(range_bw,mean_nmi_SDP_proj,std_nmi_SDP_proj,'-bo','DisplayName','mean SDP + proj')
ylim([0 1])
errorbar(range_bw,mean_nmi_DM_proj,std_nmi_DM_proj,'-rs','DisplayName','mean DM + proj')
ylim([0 1])
legend('nmi kmeans raw data','nmi SDP+proj','nmi DM+proj')

matlab2tikz('Figures/mean_proj_twomoons_gaussians_benchmark.tikz')
saveas(gcf,'Figures/mean_proj_twomoons_gaussians_benchmark','epsc')


figure;
plot(range_bw, eigs_SDP,'.-','MarkerSize',15)
matlab2tikz('Figures/eigs_SDP_twomoons_gaussians_benchmark.tikz')
saveas(gcf,'Figures/eigs_SDP_twomoons_gaussians_benchmark','epsc')

figure;
plot(range_bw, eigs_DM,'.-','MarkerSize',15)

matlab2tikz('Figures/eigs_DM_twomoons_gaussians_benchmark.tikz')
saveas(gcf,'Figures/eigs_DM_twomoons_gaussians_benchmark','epsc')
