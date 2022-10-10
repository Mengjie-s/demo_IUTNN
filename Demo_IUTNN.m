
%% =================================================================
clc;
clear;
close all;
addpath(genpath('ToolBox'));
addpath(genpath('quality_assess'));
addpath(genpath('data'));
%%
methodname    = {'Observed','IUTNN'};
Mnum = length(methodname);
Re_tensor  =  cell(Mnum,1);
psnr       =  zeros(Mnum,1);
ssim       =  zeros(Mnum,1);
fsim       =  zeros(Mnum,1);
time       =  zeros(Mnum,1);

%% Load initial data
load('fake_and_real_beers.mat')%simu_indian
 X=img;
if max(X(:))>1
    X = my_normalized(X);
end
%% Sampling with random position
sample_ratio = 0.2;
fprintf('=== The sample ratio is %4.2f ===\n', sample_ratio);
Y_tensorT = X;
Ndim      = ndims(Y_tensorT);
Nway      = size(Y_tensorT);
Omega=gen_W(size(X),1-sample_ratio);     
Y_tensor0=X.*Omega;
Omega1     = find(rand(prod(Nway),1)<sample_ratio);
Y_tensor1 = zeros(Nway);
Y_tensor1(Omega1) = Y_tensorT(Omega1);
%%
i  = 1;
Re_tensor{i} = Y_tensor0;
[psnr(i), ssim(i), fsim(i)] = quality(Y_tensorT*255, Re_tensor{i}*255);
enList = 1;
%% Use IUTNN
i=i+1;
    opts=[];
    alpha = zeros(Ndim,Ndim);
            for ii=1:Ndim-1
                for jj=ii+1:Ndim
                    alpha(ii,jj)=1;
                end
            end
    opts.alpha=alpha/sum(alpha(:));
    opts.tol = 1e-4;
    opts.maxit = 500;
    opts.rho = 1.1;
    opts.beta = opts.alpha/100;
    opts.max_beta = 1e10*ones(Ndim,Ndim);
    fprintf('\n');
    disp(['performing ',methodname{i}, ' ... ']);
     t0= tic;
tic;
    [Re_tensor{i},~]=LRTC_IUTNN(Y_tensor0,Omega,opts);
    time(i)= toc(t0);
toc;
    [psnr(i), ssim(i), fsim(i)] = quality(Y_tensorT*255, Re_tensor{i}*255);
    enList = [enList,i];

%% Show result
fprintf('\n');
fprintf('================== Result =====================\n');
fprintf(' %8.8s    %5.4s    %5.4s    %5.4s   %5.4s  \n','method','PSNR', 'SSIM', 'FSIM','time');
for i = 1:length(enList)
    fprintf(' %8.8s    %5.3f    %5.3f    %5.3f   %5.3f\n',...
        methodname{enList(i)},psnr(enList(i)), ssim(enList(i)), fsim(enList(i)),time(i));
end
fprintf('================== Result =====================\n');
figure,
showMSIResult(Re_tensor,Y_tensorT,min(Y_tensorT(:)),max(Y_tensorT(:)),methodname,enList,1,Nway(3))



