import pandas as pd

from factors.construction.factor_calculation._base_high_freq_factor import _BaseHighFreqFactor


class ClosingAuctionVolumeRatio(_BaseHighFreqFactor):
    factor_name = "ClosingAuctionVolumeRatio"
    factor_type = "liquidity"
    report_factor_code = "liq_lastCallR"
    calculation_logic = "14:57至15:00收盘集合竞价成交量占全天成交量。"

    def __init__(self, minutes, ticks=None, **kwargs):
        super().__init__(minutes)
        self.ticks = minutes if ticks is None else ticks

    def calculate(self):
        ticks = self.ticks.copy()
        ticks["trade_time"] = pd.to_datetime(ticks["trade_time"])
        ticks = ticks.sort_values(["code", "date", "trade_time"])
        def value(group):
            mask = group["trade_time"].dt.time.between(pd.Timestamp("14:57").time(), pd.Timestamp("15:00").time())
            selected = pd.to_numeric(group.loc[mask, "volume"], errors="coerce").sum()
            if "ratio" == "raw":
                return selected
            total = pd.to_numeric(group["volume"], errors="coerce").sum()
            return selected / total if total else float("nan")
        return ticks.groupby(["code", "date"]).apply(value).rename(self.factor_name).reset_index()
