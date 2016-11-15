function F = SimTaxModelFOC(x, wage, tax, trans, sigg, cchi)
    cons = x(1);
    hours = x(2);
    F(1) = (wage * hours * (1 - tax) + trans)^(-sigg) * wage * (1 - tax) - ...
           hours^cchi;
    F(2) = wage * hours * (1 - tax) + trans - cons;
end
