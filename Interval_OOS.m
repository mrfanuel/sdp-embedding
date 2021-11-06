%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%Code for Embedding of the interval [-1,1]
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
clear; close all;
addpath('Data')
addpath('Algo')
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

bw = 0.1;

%% sampled interval
n_tot = 15;
dx = 2/n_tot;
x = ((-1+dx/2):dx: (1-dx/2))';
N = length(x);

%% diffusion kernel matrix
d = pdist2(x,x);
k = exp(-d.^2/bw^2);
deg = dx*sum(k,2);

v0 = sqrt(deg/sum(dx*deg));

k_norm = diag(1./sqrt(deg))*k*diag(1./sqrt(deg));
K = (k_norm-v0*v0');

%% SDP solution

diagonal = diag(K);

n_it = 50000; % maximal number of iterations
tol = 1e-09; % tolerance on relative difference between 2 iterates

% Initialization
r = 10;
H0 = rand(N,r)-0.5; 
H0 = diag(sqrt(diagonal))*H0./sqrt(sum(H0.^2,2));

% Iterations
[H,~] = ProjPowerIterations(K,H0,diagonal,n_it,tol);
%[Answer,normgradient] = IsLocalMaximum(H,Q,diagonal); % checking optimality 

% SVD 
[V,L] = svd(H);
l = diag(L);
disp('Singular values of the solution')
disp(l)


%% kernel matrix
B = H*H';

%% Plotting
u0 =  l(1)*V(:,1);
u1 =  l(2)*V(:,2);
%figure;scatter(u0,u1,[],'o','filled'); title('SDP embedding')


u = [u0 u1];

%% out of sample 

n_oos = 100; % number of points
l_n = 2/n_oos; % interdistance

x_oos = ((-1+l_n/2):l_n: (1-l_n/2))';

%% extension of the diffusion kernel
d_oos_x = pdist2(x_oos,x);
k_oos = exp(-d_oos_x.^2/bw^2); 
deg_oos = sum(k_oos,2)*dx;
k_ext_norm = diag(1./sqrt(deg_oos))*k_oos*diag(1./sqrt(deg));
v0_ext = dx*k_ext_norm*v0;

K_ext = k_ext_norm -v0_ext*v0';
nor = 1./deg_oos-deg_oos/(sum(dx*deg));
u_oos_0 = dx*K_ext*u;
M = sum(u_oos_0.^2,2);
u_oos = diag(sqrt(nor))*u_oos_0./sqrt(M);
    
B_oos =   u_oos*u_oos';

figure; plot(x,u0,'.'); hold on; plot(x,u1,'.'); 
xlabel('x'); ylabel('\chi_1 and \chi_2')
place = 'Figures/sparse_interval_eig.png';
saveas(gcf,place);
close all;  

figure; scatter(x,u0,[],'ok','filled'); hold on; scatter(x,u1,[],'ok','filled');
scatter(x_oos,u_oos(:,1),[],'.b');
scatter(x_oos,u_oos(:,2),[],'.r');
title('Out-of-sample extension')    
xlabel('$x$','Interpreter','latex','FontSize',15); ylabel('$\chi_\ell(x)$','Interpreter','latex','FontSize',15)
place = 'Figures/oos_sparse_interval_eig.png';
saveas(gcf,place);
close all;  

figure;scatter(u0,u1,[],'ok','filled');hold on
scatter(u_oos(:,1),u_oos(:,2),[],'.'); title('SDP embedding')        
xlabel('$\chi_1(x)$','Interpreter','latex','FontSize',15); ylabel('$\chi_2(x)$','Interpreter','latex','FontSize',15)
place = 'Figures/oos_sparse_interval_embedding.png';
saveas(gcf,place);
close all;  

y = eigs(dx*K,1);
disp ('Largest eig')
disp(y)
disp ('Objective')
disp (dx*dx*trace(B*K)/trace(dx*K))
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%       Now dense sampling
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% interval [-1,1]
n_tot = 2000;
dx = 2/n_tot;
dsurf = dx*dx;

x_dense = (-1+dx/2):dx: (1-dx/2);

d = zeros(length(x_dense),length(x_dense));

for i = 1:length(x_dense)
    for j=1:length(x_dense)
        d(i,j) = abs(x_dense(i)-x_dense(j));
    end
end
dist = triu(d-diag(diag(d))); 
[el,~] = find(dist~=0);

k = exp(-d.^2/bw^2);
deg = dx*sum(k,2);

%diffusion kernel matrix
v0 = sqrt(deg/sum(dx*deg));

k_norm = diag(1./sqrt(deg))*k*diag(1./sqrt(deg));
K = (k_norm-v0*v0');

diagonal = diag(K);

Q = K;
n_it = 50000;
tol = 1e-9;
% Initialization
    r = 10;N = length(x_dense);
    H0 = rand(N,r)-0.5; H0 = diag(sqrt(diagonal))*H0./sqrt(sum(H0.^2,2));

% Iterations
    [H,dist] = ProjPowerIterations(Q,H0,diagonal,n_it,tol);
    [Answer,normgradient] = IsLocalMaximum(H,Q,diagonal);
    [V,L] = svd(H);

disp('Eigenvalues of PCA ...')
    l = diag(L);[id,~] = find(l>1e-05);
    nb_nnz = nnz(l>1e-05);
    disp(nb_nnz)
p = deg/sum(deg);
B_dense = H*H';

B_test = sqrt(p)*sqrt(p)'+B_dense;


%% Plotting
disp('Plotting ...')
if l(3)/length(H)>1e-05
    u0_dense =  l(1)*V(:,1);
    u1_dense =  l(2)*V(:,2);
    u2_dense =  l(3)*V(:,3);
    figure;scatter3(u0_dense,u1_dense,u2_dense,[],'.'); title('SDP embedding')

elseif l(2)/length(H)>1e-05
    u0_dense =  l(1)*V(:,1);
    u1_dense =  l(2)*V(:,2);
    figure;scatter(u0_dense,u1_dense,[],'.'); title('SDP embedding')        

else
        disp('Only rank 1')
end


disp ('Objective denser sampling')
disp (dx*dx*trace(B_dense*K)/trace(dx*K))

figure; plot(x_dense,u0_dense,'.'); hold on; plot(x_dense,u1_dense,'.'); 
xlabel('x'); ylabel('\chi_1 and \chi_2')
place = 'Figures/dense_interval_embedding.png';
saveas(gcf,place);
close all;  

figure; plot(x,u0,'*'); hold on; plot(x,u1,'*'); 
plot(x_dense,u0_dense,'.'); hold on; plot(x_dense,u1_dense,'.'); 
plot(x_oos,u_oos(:,1),'.');
plot(x_oos,u_oos(:,2),'.');


xlabel('x'); ylabel('\chi_1 and \chi_2')
title('Comparison samplings')
place = 'Figures/oos_dense_interval_eig.png';
saveas(gcf,place);
close all;  

y = eigs(dx*K,1);
disp ('Largest eig')
disp(y)

[X0,Y0] = meshgrid(x,x);

x = [];
y = [];
z = [];
for i =1:length(X0)
    for j =1:length(X0)
        x = [x;X0(i,j)];
        y = [y;Y0(i,j)];
        z = [z;B(i,j)];
    end
end
figure;
scatter3(x,y,z,'o','MarkerFaceColor','r'); hold on;


B_n = u_oos*u_oos';
[X0,Y0] = meshgrid(x_oos,x_oos);

x = [];
y = [];
z = [];
for i =1:length(X0)
    for j =1:length(X0)
        x = [x;X0(i,j)];
        y = [y;Y0(i,j)];
        z = [z;B_n(i,j)];
    end
end
%scatter3(x,y,z,'.'); 
surf(X0,Y0,B_n)

