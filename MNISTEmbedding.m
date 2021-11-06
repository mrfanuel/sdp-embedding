clear; close all;
addpath('Data')
addpath('Algo')

%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Embedding full MNIST
%%%%%%%%%%%%%%%%%%%%%%%%%%%

%%  Training
X_digits = (loadMNISTImages('train-images.idx3-ubyte'))';
Y_digits = (loadMNISTLabels('train-labels.idx1-ubyte'));
N = length(Y_digits);

% shuffle
ind = randperm(N,N);
N = length(ind);
X_digits = X_digits(ind,:);
Y_digits = Y_digits(ind,:);

% identify classes

id0 = find(Y_digits==0);
id1 = find(Y_digits==1);
id2 = find(Y_digits==2);
id3 = find(Y_digits==3);
id4 = find(Y_digits==4);
id5 = find(Y_digits==5);
id6 = find(Y_digits==6);
id7 = find(Y_digits==7);
id8 = find(Y_digits==8);
id9 = find(Y_digits==9);

X = X_digits([id0;id3;id7],:);
Y = Y_digits([id0;id3;id7],:);
place = 'Figures/mnist037.eps';

%%
d = pdist2(X,X);

%%
bw = 3
k = exp(-d.^2/bw^2);
deg = sum(k,2);
v0 = sqrt(deg/sum(deg));

k_norm = diag(1./sqrt(deg))*k*diag(1./sqrt(deg));
K = (k_norm-v0*v0');% diffusion kernel matrix


% SDP solution
N = size(X,1);
diagonal = diag(K);

n_it = 5000; % maximal number of iterations
tol = 1e-09; % tolerance on relative difference between 2 iterates

% Initialization
r = 20;
H0 = rand(N,r)-0.5; 
H0 = diag(sqrt(diagonal))*H0./sqrt(sum(H0.^2,2));

% Iterations
[H,~] = ProjPowerIterations(K,H0,diagonal,n_it,tol);
%[Answer,normgradient] = IsLocalMaximum(H,Q,diagonal); % checking optimality 

% SVD 
[V,L] = svd(H);
l = diag(L);
fprintf('Eigenvalues of PCA: %d \n',nnz(l>1e-08))

% kernel matrix
rho = H*H';

% Plotting
u0 =  l(1)*V(:,1);
u1 =  l(2)*V(:,2);
[X_c,Y_c] = eigs(k_norm);

%
s = mean(deg)*deg/(sum(deg));
figure;scatter(u0,u1,[],Y,'o','filled'); title('SDP embedding')       
colorbar; colormap jet

xl = get(gca,'XLabel');
xlFontSize = get(xl,'FontSize');
xAX = get(gca,'XAxis');
yAX = get(gca,'YAxis');
zAX = get(gca,'ZAxis');

set(xAX,'FontSize', 15)
set(yAX, 'FontSize', 15);
set(zAX, 'FontSize', 15);
saveas(gcf,place,'epsc');

%

figure; 
scatter(X_c(:,2),X_c(:,3),[],Y,'.')
colorbar; colormap jet
 title('Diffusion embedding')       