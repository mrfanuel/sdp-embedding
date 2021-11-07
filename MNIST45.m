%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%Code for replicating the SDP embedding of MNIST digits 1 and 4
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
clear;close all;

addpath(genpath('Data'))  
addpath(genpath('Algo'))  
%%%%%%%%%%%%%%%%%%%%%%%%%% Loading training data of MNIST %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

images = loadMNISTImages('train-images.idx3-ubyte');
labels = loadMNISTLabels('train-labels.idx1-ubyte');
id4 = find(labels==1);x4 = images(:,id4);
id5 = find(labels==4);x5 = images(:,id5);
x = [x4 x5]';
truth45 = [labels(id4);labels(id5)];
N_tot = length(truth45);
%%%%%%%%%%%%%%%%%%%%%%%%%% Computing Kernel matrices %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
bw = 10;

n_it = 50000; % max number of iterations
tol = 1e-09; % stopping criterion
r = 100;    

% selecting a subset of the digits to compute the embedding
n_train = floor(0.3*N_tot);
id_train =  datasample(1:N_tot,n_train,'Replace',false);

[V_SDP,~,sqrt_eigenvalues_SDP,~,deg] = embed(x,id_train,bw,r,n_it,tol);

v0 = sqrt(deg/sum(deg));
%%%%%%%%%%%%%%%%%%%%%%%%% % Embedding of the training set %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%


figure;scatter(V_SDP(:,1),V_SDP(:,2),[],truth45(id_train),'.'); title('SDP embedding') 
colormap jet;
%place = '/Figures/mnist45_SDP_embeding.png';
%saveas(gcf,place)
close all;
%%%%%%%%%%%%%%%%%%%%%%%%%% Out-of-sample extension %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
dim = 2;
id_oos = setdiff(1:N_tot,id_train);
n_oos = length(id_oos);

% initialization
u_train = V_SDP;
u_oos = zeros(n_oos,dim);
d = zeros(n_train,1);

for i=1:n_oos
    x_oos = x(id_oos(i),:);
    for j=1:n_train
        d(j) = norm(x_oos-x(id_train(j),:),2);
    end
    k_oos = exp(-d.^2/bw^2)'; 
    deg_oos = sum(k_oos,2);
    k_oos_norm = (1/sqrt(deg_oos))*k_oos*diag(1./sqrt(deg));
    v0_oos = k_oos_norm*v0;

    K_oos = k_oos_norm -v0_oos*v0';
    nor = 1/deg_oos-deg_oos/(sum(deg));
    u_oos0 = K_oos*u_train;
    M = sum(u_oos0.^2);
    u_oos(i,:) = sqrt(nor)*u_oos0/sqrt(M);
end    


%%%%%%%%%%%%%%%%%%%%%%%%%% Plotting %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
figure;
scatter(V_SDP(:,1),V_SDP(:,2),3,truth45(id_train),'o','filled');
colormap jet;hold on
xlim([-0.03 0.04])
ylim([-0.03 0.03])
figure;
s = scatter(u_oos(:,1),u_oos(:,2),2,truth45(id_oos),'s','filled');
colormap jet;
hold off
xlim([-0.03 0.04])
ylim([-0.03 0.03])

xl = get(gca,'XLabel');
xlFontSize = get(xl,'FontSize');
xAX = get(gca,'XAxis');
yAX = get(gca,'YAxis');
zAX = get(gca,'ZAxis');

set(xAX,'FontSize', 15)
set(yAX, 'FontSize', 15);
set(zAX, 'FontSize', 15);

place = 'Figures/mnist45_oos.png';
saveas(gcf,place);
%close all;

figure;scatter(V_SDP(:,1),V_SDP(:,2),3,truth45(id_train),'o','filled');colormap jet;hold on
xl = get(gca,'XLabel');
xlFontSize = get(xl,'FontSize');
xAX = get(gca,'XAxis');
yAX = get(gca,'YAxis');
zAX = get(gca,'ZAxis');

set(xAX,'FontSize', 15)
set(yAX, 'FontSize', 15);
set(zAX, 'FontSize', 15);
place = 'Figures/mnist45_training.png';
saveas(gcf,place);
%close all;


%%%%%%%%%%%%%%%%%%%%%%%%%% Training of nnb classifier %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%Mdl = fitcsvm(u_train,truth45(id_train));
Mdl = fitcknn(u_train,truth45(id_train),'NumNeighbors',5,'Standardize',1)

Class = predict(Mdl, u_oos);
disp('accuracy')
disp(1-nnz(Class-truth45(id_oos))/length(Class))


%x1range = min(u_oos(:,1)):.0001:max(u_oos(:,1));
%x2range = min(u_oos(:,2)):.0001:max(u_oos(:,2));
%[xx1, xx2] = meshgrid(x1range,x2range);
%XGrid = [xx1(:) xx2(:)];
%predicted = predict(Mdl,XGrid);
%figure;
%gscatter(xx1(:), xx2(:), predicted,'gray');colormap jet;hold on
%scatter(u0,u1,3,truth45(id_train),'o','filled');colormap jet

