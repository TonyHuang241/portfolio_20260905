import numpy as np
import pandas as pd

from factors.construction.factor_calculation._base_high_freq_factor import _BaseHighFreqFactor


class RollingHighLowRSquaredMean(_BaseHighFreqFactor):
    factor_name = "RollingHighLowRSquaredMean"
    factor_type = "momentum_reversal"
    report_factor_code = "mmt_ols_corr_square_mean"
    calculation_logic = "滚动50根分钟K线最高价与最低价相关系数平方的日内均值。"
    window = 50

    @classmethod
    def _components(cls, group):
        high = pd.to_numeric(group["high"], errors="coerce")
        low = pd.to_numeric(group["low"], errors="coerce")
        rolling_low = low.rolling(cls.window, min_periods=cls.window)
        beta = rolling_low.cov(high, ddof=0) / rolling_low.var(ddof=0)
        corr = rolling_low.corr(high)
        values = pd.DataFrame({
            "beta": beta,
            "corr": corr,
            "r_squared": corr * corr,
            "qrs": beta * corr * corr,
        })
        return values.replace([np.inf, -np.inf], np.nan).dropna(subset=["beta", "corr"])

    def calculate(self):
        minutes = self.minutes.copy()
        minutes["trade_time"] = pd.to_datetime(minutes["trade_time"])
        minutes = minutes.sort_values(["code", "date", "trade_time"])
        def value(group):
            values = self._components(group)
            if values.empty:
                return float("nan")
            if "r_squared" == "beta_zscore_last":
                std = values["beta"].std(ddof=1)
                return (values["beta"].iloc[-1] - values["beta"].mean()) / std if std else float("nan")
            return values["r_squared"].mean()
        return minutes.groupby(["code", "date"]).apply(value).rename(self.factor_name).reset_index()
