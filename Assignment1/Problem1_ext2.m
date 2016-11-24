%% Exercise 1

% Initialize stuff
w = 1;
siggmaGrid = [.1, 1, 4];
cchi = 1;
TaxRates = linspace(0.01, 0.99, 99);

% Create 3 dimensional array for policies welfare and transfers. For all
% of those the first dimension corresponds to entries for a fixed siggma,
% the second dimension is for the policy variable, the third dimension
% captures variation in the Frisch labour elasticity
policies = nan(length(TaxRates), 2, length(siggmaGrid));
trans_arr = nan(length(TaxRates), 1, length(siggmaGrid));
welf_arr = nan(length(TaxRates), 1, length(siggmaGrid));

% loop over Grid of Frisch elasticities
count = 1;
for sig = siggmaGrid
    policies(:, :, count) = getPolicy2(TaxRates, ...
        @(x, tau) SimTaxModelFOC(x, w, tau, sig, cchi), w);
    trans_arr(:, :, count) = w * policies(:,2,count) .* TaxRates';
    if sig == 1
        welf_arr(:, :, count) = log(policies(:,1,count)) - ...
            (policies(:, 2, count).^(1 + cchi)) / (1 + cchi);
    else
        welf_arr(:, :, count) = (policies(:,1,count).^(1 - sig) - 1) / (1 - sig) - ...
            (policies(:, 2, count).^(1 + cchi)) / (1 + cchi);
    end
    count = count + 1;
end

fig = figure();
for idx = 1:length(siggmaGrid)
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
leg = legend('$\sigma = 0.1$', '$\sigma = 1$', '$\sigma = 4$',...
    'Location', 'southwest');
set(leg, 'Interpreter', 'latex')
subplot(2, 2, 2)
title('Hours worked')
subplot(2, 2, 3)
title('Transfers')
subplot(2, 2, 4)
title('Welfare')
