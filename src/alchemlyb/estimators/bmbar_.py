from typing import Literal

import numpy as np
import pandas as pd
from bayesmbar import BayesMBAR as _BayesMBAR
from sklearn.base import BaseEstimator

from alchemlyb.estimators.base import _EstimatorMixOut


class BayesMBAR(BaseEstimator, _EstimatorMixOut):
    def __init__(
        self,
        prior: Literal["uniform", "normal"] = "uniform",
        mean: Literal["constant", "linear", "quadratic"] = "constant",
        kernel: Literal["SE", "Matern52", "Matern32", "RQ"] = "SE",
        state_cv: np.ndarray = None,
        sample_size: int = 1000,
        warmup_steps: int = 500,
        optimize_steps: int = 10000,
        verbose: bool = True,
        random_seed: int = 0,
        method: Literal["Newton", "L-BFGS-B"] = "Newton",
    ):
        self.prior = prior
        self.mean = mean
        self.kernel = kernel
        self.state_cv = state_cv
        self.sample_size = sample_size
        self.warmup_steps = warmup_steps
        self.optimize_steps = optimize_steps
        self.verbose = verbose
        self.random_seed = random_seed
        self.method = method
        self._bmbar = None
        self._delta_f_mode_ = None

    def fit(self, u_nk):
        u_nk = u_nk.sort_index(level=u_nk.index.names[1:])

        groups = u_nk.groupby(level=u_nk.index.names[1:])
        num_conf = np.array([
            (
                len(groups.get_group(i if isinstance(i, tuple) else (i,)))
                if i in groups.groups
                else 0
            )
            for i in u_nk.columns
        ])
        self._states_ = u_nk.columns.values.tolist()
        u = u_nk.T.to_numpy()
        self._bmbar = _BayesMBAR(
            u,
            num_conf,
            prior=self.prior,
            mean=self.mean,
            state_cv=self.state_cv,
            kernel=self.kernel,
            sample_size=self.sample_size,
            warmup_steps=self.warmup_steps,
            optimize_steps=self.optimize_steps,
            random_seed=self.random_seed,
            verbose=self.verbose,
        )
        delta_f_ = self._bmbar.DeltaF_mean
        delta_f_ -= delta_f_[0,0]
        self._delta_f_ = pd.DataFrame(
            delta_f_, columns=self._states_, index=self._states_
        )
        delta_f_mode_ = self._bmbar.DeltaF_mode
        delta_f_mode_ -= delta_f_mode_[0,0]
        self._delta_f_mode_ = pd.DataFrame(
            delta_f_mode_, columns=self._states_, index=self._states_
        )
        self._d_delta_f_ = pd.DataFrame(
            self._bmbar.DeltaF_std, columns=self._states_, index=self._states_
        )
        self._delta_f_.attrs = u_nk.attrs
        self._delta_f_mode_.attrs = u_nk.attrs
        self._d_delta_f_.attrs = u_nk.attrs
        return self

    @property
    def delta_f_mode_(self):
        return self._delta_f_mode_