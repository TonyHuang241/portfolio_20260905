import pandas as pd

from factors.construction.factor_calculation._base_high_freq_factor import _BaseHighFreqFactor


class MinuteReturnSkewKurtosisRatio(_BaseHighFreqFactor):
    factor_name = "MinuteReturnSkewKurtosisRatio"
    factor_type = "higher_order_moments"
    report_factor_code = "shape_skratio"
    calculation_logic = "日内分钟收益率偏度与超额峰度的比值。"

    def calculate(self):
        minutes = self.minutes.copy()
        minutes["trade_time"] = pd.to_datetime(minutes["trade_time"])
        minutes = minutes.sort_values(["code", "date", "trade_time"])
        def value(group):
            values = pd.to_numeric(group["close"], errors="coerce").pct_change(fill_method=None)
            values.iloc[0] = group["close"].iloc[0] / group["open"].iloc[0] - 1
            kurtosis = values.kurt()
            return values.skew() / kurtosis if kurtosis else float("nan")
        return minutes.groupby(["code", "date"]).apply(value).rename(self.factor_name).reset_index()
