import pandas as pd

from factors.construction.factor_calculation._base_high_freq_factor import _BaseHighFreqFactor


class AfternoonMomentum(_BaseHighFreqFactor):
    factor_name = "AfternoonMomentum"
    factor_type = "momentum_reversal"
    report_factor_code = "mmt_pm"
    calculation_logic = "下午时段首根开盘价到末根收盘价的收益率。"

    def calculate(self):
        minutes = self.minutes.copy()
        minutes["trade_time"] = pd.to_datetime(minutes["trade_time"])
        minutes = minutes.loc[minutes["trade_time"].dt.time.between(pd.Timestamp("13:00").time(), pd.Timestamp("15:00").time())]
        result = minutes.sort_values(["code", "date", "trade_time"]).groupby(["code", "date"], as_index=False).agg(start=("open", "first"), end=("close", "last"))
        result[self.factor_name] = result["end"] / result["start"] - 1
        return result[["code", "date", self.factor_name]]
