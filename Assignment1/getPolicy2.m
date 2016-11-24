function policy = getPolicy_Ex2(TaxRates, FOC, wage)
% Calculate the policy function of the housholds given the First order 
% condition of the households optimization problem over a set of taxrates.

policy = nan(length(TaxRates), 2);
transfers = nan(length(TaxRates), 1);
idx = 1;
for tau = TaxRates
    % Use fsolve to find root of the system of FOCs over the choice of
    % consumption and hours
    % Create new function where FOCs are fixed at initialized values
    policy(idx, :) = fsolve(@(x) FOC(x, tau), [.5, .5]);
    idx = idx + 1;
end
end
