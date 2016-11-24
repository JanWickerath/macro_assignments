function policy = getPolicy(FOC, TaxRates, wage, siggma, frisch)
% Calculate the policy function of the housholds given the First order 
% condition of the households optimization problem over a set of taxrates.

policy = nan(length(TaxRates), 2);
transfers = nan(length(TaxRates), 1);
idx = 1;
for tau = TaxRates
    % Use fsolve to find root of the system of FOCs over the choice of
    % consumption and hours
    % Create new function where FOCs are fixed at initialized values
    focForTau = @(x) SimTaxModelFOC(x, w, tau, siggma, cchi);
    policy(idx, :) = fsolve(focForTau, [.5, .5]);
    transfers(idx, 1) = w * policy(idx, 2) * tau;
    idx = idx + 1;
end

transfers = w * policy(:,2) .* TaxRates';
welfare = (policy(:,1).^(1 - siggma) - 1) / (1 - siggma) - ...
    (policy(:, 2).^(1 + cchi)) / (1 + cchi);

% Plot results
fig = figure();
subplot(2, 2, 1)
plot(TaxRates, policy(:, 1))
title('Consumption')
subplot(2, 2, 2)
plot(TaxRates, policy(:, 2))
title('Hours worked')
subplot(2, 2, 3)
plot(TaxRates, transfers)
title('Transfers')
subplot(2, 2, 4)
plot(TaxRates, welfare)
title('Welfare')
end
