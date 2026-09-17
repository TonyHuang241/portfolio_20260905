import pandas as pd

from factors.construction.factor_calculation._base_high_freq_factor import _BaseHighFreqFactor


class MinuteVolumeShareSkewKurtosisRatio(_BaseHighFreqFactor):
    factor_name = "MinuteVolumeShareSkewKurtosisRatio"
    factor_type = "higher_order_moments"
    report_factor_code = "shape_skratioVol"
    calculation_logic = "分钟成交量占比序列偏度与超额峰度的比值。"

    def calculate(self):
        minutes = self.minutes.copy()
        minutes["trade_time"] = pd.to_datetime(minutes["trade_time"])
        minutes = minutes.sort_values(["code", "date", "trade_time"])
        def value(group):
            values = pd.to_numeric(group["volume"], errors="coerce")
            values = values / values.sum()
            kurtosis = values.kurt()
            return values.skew() / kurtosis if kurtosis else float("nan")
        return minutes.groupby(["code", "date"]).apply(value).rename(self.factor_name).reset_index()
