% Two moons generator inspired by
%  BISPL, KAIST, Jaejun Yoo, e-mail: jaejun2004@gmail.com
% adapted by M. Fanuel

function [X,Y] = gen_two_moons(center_up,center_down,radius,nb_sample,noise_eps)

theta = linspace(0,pi,nb_sample);
noise = rand(1,nb_sample)*noise_eps;
semi_up = [(radius+noise).*cos(theta) + center_up(1);(radius+noise).*sin(theta) + center_up(2)];
semi_down = [(radius+noise).*cos(-1*theta) + center_down(1) ; (radius+noise).*sin(-1*theta) + center_down(2) ] ;


X = [semi_up,semi_down]';
Y = [ones(length(semi_up),1);-1*ones(length(semi_down),1)];
end

