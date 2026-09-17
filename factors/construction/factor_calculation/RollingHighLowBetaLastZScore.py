import numpy as np
import pandas as pd

from factors.construction.factor_calculation._base_high_freq_factor import _BaseHighFreqFactor


class RollingHighLowBetaLastZScore(_BaseHighFreqFactor):
    factor_name = "RollingHighLowBetaLastZScore"
    factor_type = "momentum_reversal"
    report_factor_code = "mmt_ols_beta_zscore_last"
    calculation_logic = "最后一个滚动50分钟beta相对当日beta序列的标准分。"
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
            if "beta_zscore_last" == "beta_zscore_last":
                std = values["beta"].std(ddof=1)
                return (values["beta"].iloc[-1] - values["beta"].mean()) / std if std else float("nan")
            return values["beta_zscore_last"].mean()
        return minutes.groupby(["code", "date"]).apply(value).rename(self.factor_name).reset_index()
