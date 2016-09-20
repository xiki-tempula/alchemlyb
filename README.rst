alchemlyb: data munging and analysis library for alchemical free energy calculations
====================================================================================

**Warning**: This library is young. It is **not** API stable. It is a
nucleation point. By all means use and help improve it, but note that it will
change with time.

**alchemlyb** is an attempt to make alchemical free energy calculations easier
to do leveraging the full power and flexibility of the PyData stack. It
includes: 

1. functions for extracting raw data from output files of common
molecular dynamics engines such as GROMACS [Abraham2015]_. 

2. functions for obtaining free energies directly from this data, using
best-practices approaches for Multistate Bennett Acceptance Ratio (MBAR)
[Shirts2008]_ and thermodynamic integration (TI).

This library currently centers around the use of `MDSynthesis
<http://mdsynthesis.readthedocs.org>`_ at the moment for memory efficiency,
persistence, and code simplicity, though this may change in the future.

.. [Abraham2015] Abraham, M.J., Murtola, T., Schulz, R., Páll, S., Smith, J.C.,
    Hess, B., and Lindahl, E. (2015). GROMACS: High performance molecular
    simulations through multi-level parallelism from laptops to supercomputers.
    SoftwareX 1–2, 19–25.

.. [Shirts2008] Shirts, M.R., and Chodera, J.D. (2008). Statistically optimal
    analysis of samples from multiple equilibrium states. The Journal of Chemical
    Physics 129, 124105.
