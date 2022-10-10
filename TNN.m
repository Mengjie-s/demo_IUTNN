function [L E k eta aaa ] = TNN(X,Omega,dim,opts)

%% Robust tensor completion using transformed tensor singular value decomposition
% by Guangjing Song  Michael K. Ng  Xiongjun Zhang, Numerical Linear Algebra with Applications,
%  27(3):e2299, 2020.
% sGS-ADMM for robust tensor completion
%%

if ~exist('opts', 'var')
    opts = [];
end    
if isfield(opts, 'tol');         tol = opts.tol;              end
if isfield(opts, 'beta');        beta = opts.beta;            end
% if isfield(opts, 'beta2');        beta2 = opts.beta2;            end
if isfield(opts, 'MaxIte');      MaxIte = opts.MaxIte;        end
if isfield(opts, 'Z0');          Z0 = opts.Z0;                end
if isfield(opts, 'X0');          L0 = opts.X0;                end
if isfield(opts, 'E0');          E0 = opts.E0;                end
if isfield(opts, 'mu');          mu = opts.mu;                end
if isfield(opts, 'Omega');       Omega = opts.Omega;          end
if isfield(opts, 'dim');         dim = opts.dim;              end
if isfield(opts, 'gamma');       gamma = opts.gamma;          end

dim = size(X);
% X0 = zeros(dim);
% Z0 = X0;
% Y0 = X0;
BarOmega = ones(dim) - Omega;
for k = 1:MaxIte
    % updata Mbar
    Mbar = L0 + E0 - Z0/beta;
    Mbar = X.*Omega + Mbar.*BarOmega;
    
    
    % updata X
    
    L = prox_tnn(Mbar + Z0./beta - E0,1/beta);
%     L1 = dct(eye(size(Mbar + Z0./beta - E0,3)));
%     L =prox_tnn_transform(Mbar + Z0./beta - E0,L1,1/beta);
    % updata M
    M = L + E0 - Z0/beta;
    M = X.*Omega + M.*BarOmega;
    
    % updata E
    E = prox_l1(M + Z0/beta - L, mu/beta);

    

    % updata the multiplier
    
    Z = Z0 - gamma*beta*(L +  E - M);
   
    
    %% stopping criterion
     
     if k > 30
     % KKT 
     etax = L - prox_tnn(L + Z,1);
     etax = norm(etax(:))/(1 + norm(Z(:)) + norm(L(:)));
     
     etae = E - prox_l1(Z  + E, mu);
     etae = norm(etae(:))/(1 + norm(Z(:)) + norm(E(:)));
     
     etap = L + E - M;
     etap = norm(etap(:))/(1 + norm(L(:)) + norm(M(:)) + norm(E(:)));
     
     aaa = [etax, etae, etap];
     eta = max([etax, etae, etap]);
     
     if eta <= tol 
         break;
     end
     
     end
    
    %%
    L0 = L;
    Z0 = Z;
    E0 = E;
    
    
end

end
