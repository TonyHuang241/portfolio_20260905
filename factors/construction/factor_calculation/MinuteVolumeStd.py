import pandas as pd

from factors.construction.factor_calculation._base_high_freq_factor import _BaseHighFreqFactor


class MinuteVolumeStd(_BaseHighFreqFactor):
    factor_name = "MinuteVolumeStd"
    factor_type = "volatility"
    report_factor_code = "vol_volume1min"
    calculation_logic = "日内分钟成交量的样本标准差。"

    def calculate(self):
        minutes = self.minutes.copy()
        minutes["trade_time"] = pd.to_datetime(minutes["trade_time"])
        minutes = minutes.sort_values(["code", "date", "trade_time"])
        def value(group):
            values = pd.to_numeric(group["volume"], errors="coerce")
            return values.replace([float("inf"), float("-inf")], float("nan")).std(ddof=1)
        return minutes.groupby(["code", "date"]).apply(value).rename(self.factor_name).reset_index()
