%% =================================================================
% This script compares low-rank tensor completion methods
% listed as follows:
%     1. HaLRTC        Tucker decomposition based method
%     2. TNN           t-SVD based method
%     3. WSTNN         t-SVD based method
% You can:
%     1. Type 'Demo_LRTC' to to run various methods and see the pre-computed results.
%     2. Select competing methods by turn on/off the enable-bits in Demo_LRTC.m
%
% More detail can be found in [1]
% [1] Yu-Bang Zheng, Ting-Zhu Huang*, Xi-Le Zhao,Tai-Xiang Jiang,Teng-Yu Ji,and Tian-Hui Ma.
%     Tensor N-tubal rank and its convex relaxation for low-rank tensor recovery.
%
% Please make sure your data is in range [0, 1].
%
% Created by Yu-Bang Zheng £¨zhengyubang@163.com£©
% 11/19/2018

%% =================================================================
clc;
clear;
close all;
addpath(genpath('lib'));
addpath(genpath('data'));

%% Set enable bits
EN_HaLRTC     = 0;
EN_TNN        = 0;
EN_WSTNN      = 1;
methodname    = {'Observed' 'HaLRTC','TNN','ILTNN'};
Mnum = length(methodname);
Re_tensor  =  cell(Mnum,1);
psnr       =  zeros(Mnum,1);
ssim       =  zeros(Mnum,1);
fsim       =  zeros(Mnum,1);
time       =  zeros(Mnum,1);

%% Load initial data
load('fake_and_real_beers.mat')
 X=img;
% X=Data;
if max(X(:))>1
    X = my_normalized(X);
end

%% Sampling with random position
sample_ratio = 0.3;
fprintf('=== The sample ratio is %4.2f ===\n', sample_ratio);
Y_tensorT = X;
Ndim      = ndims(Y_tensorT);
Nway      = size(Y_tensorT);
% Omega     = find(rand(prod(Nway),1)<sample_ratio);
% Y_tensor0 = zeros(Nway);
% Y_tensor0(Omega) = Y_tensorT(Omega);
Omega=gen_W(size(X),1-sample_ratio);     
Y_tensor0=X.*Omega;

%%
i  = 1;
Re_tensor{i} = Y_tensor0;
[psnr(i), ssim(i), fsim(i)] = quality(Y_tensorT*255, Re_tensor{i}*255);
enList = 1;

%% Perform  algorithms
%% Use HaLRTC
i = i+1;
if EN_HaLRTC
    % initialization of the parameters
    opts=[];
    alpha=ones(Ndim,1);
    opts.alpha=alpha/sum(alpha);
    opts.tol = 1e-4;
    opts.maxit = 500;
    opts.rho = 1.1;
    opts.beta = 1e-2;
    opts.max_beta = 1e10;
    %opts.Xtrue=Y_tensorT;
    %%%%%
    fprintf('\n');
    disp(['performing ',methodname{i}, ' ... ']);
    t0= tic;
    [Re_tensor{i},~]=LRTC_HaLRTC(Y_tensor0,Omega,opts);
    time(i)= toc(t0);
    [psnr(i), ssim(i), fsim(i)] = quality(Y_tensorT*255, Re_tensor{i}*255);
    enList = [enList,i];
end

%% Use TNN
i = i+1;
if EN_TNN
    % initialization of the parameters
    opts=[];
    opts.tol = 1e-4;
    opts.maxit = 500;
    opts.rho = 1.1;
    opts.beta = 1e-2;
    opts.max_beta = 1e10;
    %opts.Xtrue=Y_tensorT;
    %%%%%
    fprintf('\n');
    disp(['performing ',methodname{i}, ' ... ']);
    t0= tic;
    [Re_tensor{i},~]=LRTC_tnn(Y_tensor0,Omega,opts);
    time(i)= toc(t0);
    [psnr(i), ssim(i), fsim(i)] = quality(Y_tensorT*255, Re_tensor{i}*255);
    enList = [enList,i];
end

%% Use ILUTNN
i = i+1;
if EN_WSTNN
    % initialization of the parameters
    % Please refer to our paper to set the parameters
    opts=[];
    alpha=[0,    1,  1;
           0,    0,    1;
           0,    0,    0]; 
    %%for Ndim-way tensors  (Ndim>3)
%     alpha = zeros(Ndim,Ndim);
%             for ii=1:Ndim-1
%                 for jj=ii+1:Ndim
%                     alpha(ii,jj)=1;
%                 end
%             end
    opts.alpha=alpha/sum(alpha(:));
    opts.tol = 1e-4;
    opts.maxit = 500;
    opts.rho = 1.1;
    opts.beta = opts.alpha/100;
    opts.max_beta = 1e10*ones(Ndim,Ndim);
    %opts.Xtrue=Y_tensorT;
    %%%%%
    fprintf('\n');
    disp(['performing ',methodname{i}, ' ... ']);
    % t0= tic;
tic;
    [Re_tensor{i},~]=LRTC_ILUTNN(Y_tensor0,Omega,opts);
   % time(i)= toc(t0);
toc;
    [psnr(i), ssim(i), fsim(i)] = quality(Y_tensorT*255, Re_tensor{i}*255);
    enList = [enList,i];
end


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



