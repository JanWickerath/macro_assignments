% Initialization
wage = 1;
muGrid = [0.3, 0.6];
sigGrid = [2, 2.5];
TaxRates = linspace(0.01, 0.99, 99);

% Create 4 dimensional array for policies welfare and transfers. For all
% of those the first dimension corresponds to entries for a fixed tax rate,
% the second dimension is for the policy variable, the third dimension
% captures variation in the sigma and the fourth dimension in the mu's.
policies = nan(length(TaxRates), 2, length(muGrid));
trans_arr = nan(length(TaxRates), 1, length(muGrid));
welf_arr = nan(length(TaxRates), 1, length(sigGrid), length(muGrid));
frisch_arr = nan(length(TaxRates), 1, length(sigGrid), length(muGrid));

% Loop over mu's and calculate policies and transfers.
countMu = 1;
for mmu = muGrid
    policies(:, :, countMu) = ...
        getPolicy_Ex2(TaxRates, @(x, tau) NonSepUtilFOC(x, wage, ...
        tau, mmu));
    trans_arr(:, :, countMu) = wage * policies(:, 2, countMu) .* TaxRates';
    countMu = countMu + 1;
end

% Loop over sigma and mu to calculate welfare levels for all combinations.
countSig = 1;
for sig = sigGrid
    countMu = 1;
    for mmu = muGrid
        welf_arr(:, :, countSig, countMu) = ...
            (policies(:, 1, countMu).^mmu .* ...
            (1 - policies(:, 2, countMu)).^(1 - mmu)).^(1 - sig) ./ (1 - sig);
        frisch_arr(:, :, countSig, countMu) = ...
            (1 - policies(:, 2, countMu)) ./ policies(:, 2, countMu) .* ...
            (1 - mmu * (1 - sig)) / sig;
        countMu = countMu + 1;
    end
    countSig = countSig + 1;
end

fig = figure();
for countSig = 1:length(sigGrid)
    for countMu = 1:length(muGrid)
        subplot(2, 2, 1)
        plot(TaxRates, policies(:, 1, countMu))
        hold on
        subplot(2, 2, 2)
        plot(TaxRates, policies(:, 2, countMu))
        hold on
        subplot(2, 2, 3)
        plot(TaxRates, trans_arr(:, 1, countMu))
        hold on
        subplot(2, 2, 4)
        plot(TaxRates, welf_arr(:, 1, countSig, countMu))
        hold on
    end
end

subplot(2, 2, 1)
title('Consumption')
leg = legend('$\mu = 0.3,~\sigma = 2$', '$\mu = 0.6,~\sigma = 2$', ...
    '$\mu = 0.3,~\sigma = 2.5$', '$\mu = 0.6,~\sigma = 2.5$', ...
    'Location', 'southwest');
set(leg, 'Interpreter', 'latex')
subplot(2, 2, 2)
title('Hours worked')
subplot(2, 2, 3)
title('Transfers')
subplot(2, 2, 4)
title('Welfare')

%% Plot Frisch elasticities

frischFig = figure();
for countSig = 1:length(sigGrid)
    for countMu = 1:length(muGrid)
        plot(TaxRates, frisch_arr(:, 1, countMu))
        hold on
    end
end
