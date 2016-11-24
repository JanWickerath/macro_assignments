function policy = getPolicy(TaxRates, wage, siggma, frisch)
% Calculate the policy function of the housholds given the First order 
% condition of the households optimization problem over a set of taxrates.

policy = nan(length(TaxRates), 2);
transfers = nan(length(TaxRates), 1);
idx = 1;
for tau = TaxRates
    % Use fsolve to find root of the system of FOCs over the choice of
    % consumption and hours
    % Create new function where FOCs are fixed at initialized values
    focForTau = @(x) SimTaxModelFOC(x, wage, tau, siggma, frisch);
    policy(idx, :) = fsolve(focForTau, [.5, .5]);
    transfers(idx, 1) = wage * policy(idx, 2) * tau;
    idx = idx + 1;
end

end
