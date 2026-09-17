import pandas as pd

from factors.construction.factor_calculation._base_high_freq_factor import _BaseHighFreqFactor


class MinuteVolumeShareSkewness(_BaseHighFreqFactor):
    factor_name = "MinuteVolumeShareSkewness"
    factor_type = "higher_order_moments"
    report_factor_code = "shape_skewVol"
    calculation_logic = "分钟成交量占全天成交量比例序列的偏度。"

    def calculate(self):
        minutes = self.minutes.copy()
        minutes["trade_time"] = pd.to_datetime(minutes["trade_time"])
        minutes = minutes.sort_values(["code", "date", "trade_time"])
        def value(group):
            values = pd.to_numeric(group["volume"], errors="coerce")
            values = values / values.sum()
            return values.skew()
        return minutes.groupby(["code", "date"]).apply(value).rename(self.factor_name).reset_index()
