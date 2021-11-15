%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%Code for Embedding of the interval [-1,1]
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
clear; close all;
addpath('../Data')
addpath('../Algo')
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

bw = 0.1;

%% sampled interval
n_tot = 15;
dx = 2/n_tot;
x = ((-1+dx/2):dx: (1-dx/2))';
N = length(x);


%% SDP solution
n_it = 50000; % maximal number of iterations
tol = 1e-09; % tolerance on relative difference between 2 iterates
r = 10;
id_train = 1:N;
nb_comp = 2;

[V_SDP,V_DM,sqrt_eigenvalues_SDP,eigenvalues_DM,deg,K] = embed(x,id_train,bw,r,n_it,tol,nb_comp,dx);
u = V_SDP;
B =   u*u';
u0 = u(:,1);
u1 = u(:,2);

%% out of sample 
n_oos = 100; % number of points
l_n = 2/n_oos; % interdistance

x_oos = ((-1+l_n/2):l_n: (1-l_n/2))';
v0 = sqrt(deg/sum(dx*deg));
u_oos = oos(x_oos,x,u,deg,v0,bw,dx); % out-of sample
B_oos =   u_oos*u_oos';

%% plotting
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

%% 
disp ('Largest eigenvalue')
disp(eigenvalues_DM)
disp ('Objective')
disp (dx*dx*trace(B*K)/trace(dx*K))
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%       Now dense sampling
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% interval [-1,1]
n_tot = 2000;
dx = 2/n_tot;
dsurf = dx*dx;

x_dense = ((-1+dx/2):dx: (1-dx/2))';
id_train_dense = 1:length(x_dense);

[V_SDP_dense,~,~,~,deg,K_dense] = embed(x_dense,id_train_dense,bw,r,n_it,tol,nb_comp,dx);
u_dense = V_SDP_dense;
B_dense =   u_dense*u_dense';
u0_dense = u_dense(:,1);
u1_dense = u_dense(:,2);


%% Plotting
figure;scatter(u0_dense,u1_dense,[],'.'); title('SDP embedding')        


disp ('Objective denser sampling')
disp (dx*dx*trace(B_dense*K_dense)/trace(dx*K_dense))

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

y = eigs(dx*K_dense,1);
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
surf(X0,Y0,B_n)

