import numpy as np
import pandas as pd

from factors.construction.factor_calculation._base_high_freq_factor import _BaseHighFreqFactor


class MinuteAmihudIlliquidity(_BaseHighFreqFactor):
    factor_name = "MinuteAmihudIlliquidity"
    factor_type = "liquidity"
    report_factor_code = "liq_amihud_1min"
    calculation_logic = "分钟绝对收益率除以分钟成交额后在日内求均值。"

    def calculate(self):
        minutes = self.minutes.copy()
        minutes["trade_time"] = pd.to_datetime(minutes["trade_time"])
        minutes = minutes.sort_values(["code", "date", "trade_time"])
        def value(group):
            returns = pd.to_numeric(group["close"], errors="coerce").pct_change(fill_method=None).abs()
            returns.iloc[0] = abs(group["close"].iloc[0] / group["open"].iloc[0] - 1)
            money = pd.to_numeric(group["money"], errors="coerce")
            return (returns / money.replace(0, np.nan)).mean()
        return minutes.groupby(["code", "date"]).apply(value).rename(self.factor_name).reset_index()
