%% Exercise 1

% Initialize stuff
w = 1;
siggma = 2;
FrischGrid = [0.6, 1, 2, 10];
TaxRates = linspace(0.01, 0.99, 99);

% Create 3 dimensional array for policies welfare and transfers. For all
% of those the first dimension corresponds to entries for a fixed siggma,
% the second dimension is for the policy variable, the third dimension
% captures variation in the Frisch labour elasticity
policies = nan(length(TaxRates), 2, length(FrischGrid));
trans_arr = nan(length(TaxRates), 1, length(FrischGrid));
welf_arr = nan(length(TaxRates), 1, length(FrischGrid));

% loop over Grid of Frisch elasticities
count = 1;
for cchi = FrischGrid
    policies(:, :, count) = getPolicy2(TaxRates, ...
        @(x, tau) SimTaxModelFOC(x, w, tau, siggma, cchi), w);
    trans_arr(:, :, count) = w * policies(:,2,count) .* TaxRates';
    welf_arr(:, :, count) = (policies(:,1,count).^(1 - siggma) - 1) / (1 - siggma) - ...
        (policies(:, 2, count).^(1 + cchi)) / (1 + cchi);
    count = count + 1;
end

fig = figure();
for idx = 1:length(FrischGrid)
    subplot(2, 2, 1)
    plot(TaxRates, policies(:, 1, idx))
    hold on
    subplot(2, 2, 2)
    plot(TaxRates, policies(:, 2, idx))
    hold on
    subplot(2, 2, 3)
    plot(TaxRates, trans_arr(:, 1, idx))
    hold on
    subplot(2, 2, 4)
    plot(TaxRates, welf_arr(:, 1, idx))
    hold on
end
subplot(2, 2, 1)
title('Consumption')
leg = legend('$\chi = 0.6$', '$\chi = 1$', '$\chi = 2$', '$\chi = 10$', ...
    'Location', 'southwest');
set(leg, 'Interpreter', 'latex')
subplot(2, 2, 2)
title('Hours worked')
subplot(2, 2, 3)
title('Transfers')
subplot(2, 2, 4)
title('Welfare')
