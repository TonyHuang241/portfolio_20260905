import pandas as pd

from factors.construction.factor_calculation._base_high_freq_factor import _BaseHighFreqFactor


class MiddleSessionMomentum(_BaseHighFreqFactor):
    factor_name = "MiddleSessionMomentum"
    factor_type = "momentum_reversal"
    report_factor_code = "mmt_between"
    calculation_logic = "剔除开盘后30分钟和收盘前30分钟后，中间时段收益率复合值。"

    @staticmethod
    def _period_return(group, start, end):
        sample = group.loc[group["trade_time"].dt.time.between(pd.Timestamp(start).time(), pd.Timestamp(end).time())]
        return sample["close"].iloc[-1] / sample["open"].iloc[0] - 1 if not sample.empty else float("nan")

    def calculate(self):
        minutes = self.minutes.copy()
        minutes["trade_time"] = pd.to_datetime(minutes["trade_time"])
        minutes = minutes.sort_values(["code", "date", "trade_time"])
        def value(group):
            morning = self._period_return(group, "10:00", "11:30")
            afternoon = self._period_return(group, "13:00", "14:30")
            return (1 + morning) * (1 + afternoon) - 1
        return minutes.groupby(["code", "date"]).apply(value).rename(self.factor_name).reset_index()
