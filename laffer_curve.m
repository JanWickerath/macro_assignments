% Initialize stuff
w = 1;
siggma = 2;
cchi = 3;
TaxRates = linspace(0.01, 0.99, 99);
transfer = .5;                  % Nowhere stated note that transfers
                                % are endogenously determined by the
                                % households labour supply, but
                                % treated as exogenous in his
                                % optimization problem (i.e. taken
                                % as given when maximizing utility).


policy = nan(length(TaxRates), 2);
idx = 1;
for tau = TaxRates
    % Use fsolve to find root of the system of FOCs over the choice of
    % consumption and hours
    % Create new function where FOCs are fixed at initialized values
    focForTau = @(x) SimTaxModelFOC(x, w, tau, transfer, siggma, cchi);
    policy(idx, :) = fsolve(focForTau, [.5, .5]);
    idx = idx + 1;
end
policy
