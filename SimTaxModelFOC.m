function F = SimTaxModelFOC(cons, hours, wage, tax, trans, sigg, xxi)
    F(1) = (wage * hours * (1 - tax) + trans)^(-sigg) * wage * (1 - tax) - ...
           hours^xxi;
    F(2) = wage * hours * (1 - tax) + trans - cons;
end
