% Initialize stuff
w = 1;
siggma = 2;
cchi = 3;
tau = .5;
transfer = .5;                  % Nowhere stated note that transfers
                                % are endogenously determined by the
                                % households labour supply, but
                                % treated as exogenous in his
                                % optimization problem (i.e. taken
                                % as given when maximizing utility).

% Use fsolve to find root of the system of FOCs over the choice of
% consumption and hours
% Create new function where FOCs are fixed at initialized values
focForTau = @(x) SimTaxModelFOC(x, w, tau, transfer, siggma, cchi);
optPol = fsolve(focForTau, [.5, .5])