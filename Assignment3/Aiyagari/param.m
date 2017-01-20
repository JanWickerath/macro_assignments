%% Parametrization.
% Set up parameters
p.alp = 0.5;
p.bet = 0.95;
p.del = 0.05;
p.sig = 2;

% Set up stochastic process.
p.rho = 0.9;
p.mu = 0;
p.var = 0.3;
p.std = sqrt(p.var*(1-p.rho));

% Set up asset grid.
p.ngrid = 100;
p.amax = 30;
p.curve = 1.5;
p.a = linspace(0, p.amax^(1/p.curve), p.ngrid).^p.curve;