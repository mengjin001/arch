import pandas as pd
from numpy import ceil, power, arange, asarray, log, pi
from numpy.linalg import lstsq
from statsmodels.tools import add_constant
from statsmodels.regression.linear_model import OLS


def _sse(y, x):
    x, y = asarray(x), asarray(y)
    b = lstsq(x, y, None)[0]
    return ((y - x @ b) ** 2).sum()


def _ic(sse, k, nobs, ic):
    llf = -nobs / 2 * (log(2 * pi) + log(sse / nobs) + 1)
    if ic == 'aic':
        penalty = 2
    elif ic == 'hqic':
        penalty = 2 * log(log(nobs))
    else:  # bic
        penalty = log(nobs)
    return -llf + k * penalty


class DynamicOLS(object):
    def __init__(self, y, x, lags=None, leads=None, common=True, max_lag=None, max_lead=None,
                 ic='aic'):
        self._y = y
        self._x = x
        self._lags = lags
        self._leads = leads
        self._common = common
        self._max_lag = max_lag
        self._max_lead = max_lead
        self._ic = ic
        self._res = None
        self._compute()

    def _compute(self):
        y, x = self._y, self._x
        k = x.shape[1]
        nobs = y.shape[0]
        delta_lead_lags = x.diff()
        max_lag = int(ceil(12. * power(nobs / 100., 1 / 4.)))
        lag_len = max_lag if self._lags is None else self._lags
        lead_len = max_lag if self._leads is None else self._leads

        lags = pd.concat([delta_lead_lags.shift(i) for i in range(1, lag_len + 1)], 1)
        lags.columns = ['D.{col}.LAG.{lag}'.format(col=col, lag=i)
                        for i in range(1, lag_len + 1)
                        for col in x]
        contemp = delta_lead_lags
        contemp.columns = ['D.{col}.LAG.0'.format(col=col) for col in x]
        leads = pd.concat([delta_lead_lags.shift(-i) for i in range(1, lead_len + 1)], 1)
        leads.columns = ['D.{col}.LEAD.{lead}'.format(col=col, lead=i)
                         for i in range(1, lead_len + 1)
                         for col in x]
        full = pd.concat([y, x, lags.iloc[:, ::-1], contemp, leads], 1).dropna()
        lhs = full.iloc[:, [0]]
        rhs = add_constant(full.iloc[:, 1:])
        base_iloc = arange(k + 1).tolist()
        sses = {}

        if self._leads is None:
            q_range = range(max_lag)
        else:
            q_range = range(self._leads, self._leads + 1)
        if self._lags is None:
            p_range = range(max_lag)
        else:
            p_range = range(self._lags, self._lags + 1)

        for p in p_range:
            for q in q_range:
                lead_lag_iloc = arange(1 + k * (1 + lag_len - p),
                                       1 + k * (1 + lag_len + 1 + q)).tolist()
                _rhs = rhs.iloc[:, base_iloc + lead_lag_iloc]
                sses[(p, q, _rhs.shape[1])] = _sse(lhs, _rhs)
        sses = pd.Series(sses)
        param_counts = sses.index.get_level_values(2)
        ics = {idx: _ic(sses[idx], k, nobs, self._ic) for k, idx in zip(param_counts, sses.index)}
        ics = pd.Series(ics)
        sel_idx = ics.idxmin()
        p, q = sel_idx[:2]
        lead_lag_iloc = arange(1 + k * (1 + lag_len - p),
                               1 + k * (1 + lag_len + 1 + q)).tolist()
        _rhs = rhs.iloc[:, base_iloc + lead_lag_iloc]
        mod = OLS(lhs, _rhs)
        res = mod.fit()
        self._res = res
        print(res.summary())

    @property
    def result(self):
        return self._res
