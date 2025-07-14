from .bar_ import BAR
from .mbar_ import MBAR
from .ti_ import TI
from .ti_gaussian_quadrature_ import TI_GQ
from .bmbar_ import BayesMBAR

FEP_ESTIMATORS = [MBAR.__name__, BAR.__name__, BayesMBAR.__name__]
TI_ESTIMATORS = [TI.__name__, TI_GQ.__name__]
