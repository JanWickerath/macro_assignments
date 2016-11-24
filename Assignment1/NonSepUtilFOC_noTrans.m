function F = NonSepUtilFOC_noTrans(x, wage, tax, mu)
% First order conditions of simple model with non-separable utility
% function and labor income taxation.
cons = x(1);
hours = x(2);
F(1) = (1 - mu) * cons / (1 - hours) - mu * wage * (1 - tax);
F(2) = cons - wage * hours * (1 - tax);
end
