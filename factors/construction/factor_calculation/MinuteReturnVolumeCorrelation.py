import numpy as np
import pandas as pd

from factors.construction.factor_calculation._base_high_freq_factor import _BaseHighFreqFactor


class MinuteReturnVolumeCorrelation(_BaseHighFreqFactor):
    factor_name = "MinuteReturnVolumeCorrelation"
    factor_type = "price_volume_correlation"
    report_factor_code = "corr_prv"
    calculation_logic = "分钟收益率与同期成交量的相关系数。"

    def calculate(self):
        minutes = self.minutes.copy()
        minutes["trade_time"] = pd.to_datetime(minutes["trade_time"])
        minutes = minutes.sort_values(["code", "date", "trade_time"])
        def value(group):
            close = pd.to_numeric(group["close"], errors="coerce")
            volume = pd.to_numeric(group["volume"], errors="coerce")
            returns = close.pct_change(fill_method=None)
            returns.iloc[0] = close.iloc[0] / group["open"].iloc[0] - 1
            volume_change = volume.pct_change(fill_method=None).replace([np.inf, -np.inf], np.nan)
            leading_volume = volume.shift(1)
            lagging_volume = volume.shift(-1)
            return returns.corr(volume)
        return minutes.groupby(["code", "date"]).apply(value).rename(self.factor_name).reset_index()
