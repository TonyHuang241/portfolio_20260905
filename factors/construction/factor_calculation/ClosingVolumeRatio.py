import pandas as pd

from factors.construction.factor_calculation._base_high_freq_factor import _BaseHighFreqFactor


class ClosingVolumeRatio(_BaseHighFreqFactor):
    factor_name = "ClosingVolumeRatio"
    factor_type = "trade_activity"
    report_factor_code = "trade_tailRatio"
    calculation_logic = "14:30至15:00成交量占全天成交量比例。"

    def __init__(self, minutes, ticks=None, **kwargs):
        super().__init__(minutes)
        self.ticks = minutes if ticks is None else ticks

    def calculate(self):
        ticks = self.ticks.copy()
        ticks["trade_time"] = pd.to_datetime(ticks["trade_time"])
        ticks = ticks.sort_values(["code", "date", "trade_time"])
        def value(group):
            volume = pd.to_numeric(group["volume"], errors="coerce")
            mask = group["trade_time"].dt.time.between(pd.Timestamp("14:30").time(), pd.Timestamp("15:00").time())
            return volume.loc[mask].sum() / volume.sum() if volume.sum() else float("nan")
        return ticks.groupby(["code", "date"]).apply(value).rename(self.factor_name).reset_index()
