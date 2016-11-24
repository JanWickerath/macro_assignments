function F = SimTaxModelFOC(x, wage, tax, sigg, cchi)
% First order conditions
cons = x(1);
hours = x(2);

% Control for log-utility case
if sigg == 1
    F(1) = hours^cchi - wage * (1 - tax) / cons;
else
    F(1) = (wage * hours)^(-sigg) * wage * (1 - tax) - hours^cchi;
end
F(2) = wage * hours - cons;
end
