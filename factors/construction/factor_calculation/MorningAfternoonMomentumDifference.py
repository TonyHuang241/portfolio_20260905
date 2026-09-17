import pandas as pd

from factors.construction.factor_calculation._base_high_freq_factor import _BaseHighFreqFactor


class MorningAfternoonMomentumDifference(_BaseHighFreqFactor):
    factor_name = "MorningAfternoonMomentumDifference"
    factor_type = "momentum_reversal"
    report_factor_code = "mmt_paratio"
    calculation_logic = "下午时段收益率减去上午时段收益率。"

    @staticmethod
    def _period_return(group, start, end):
        sample = group.loc[group["trade_time"].dt.time.between(pd.Timestamp(start).time(), pd.Timestamp(end).time())]
        return sample["close"].iloc[-1] / sample["open"].iloc[0] - 1 if not sample.empty else float("nan")

    def calculate(self):
        minutes = self.minutes.copy()
        minutes["trade_time"] = pd.to_datetime(minutes["trade_time"])
        minutes = minutes.sort_values(["code", "date", "trade_time"])
        values = minutes.groupby(["code", "date"]).apply(lambda x: self._period_return(x, "13:00", "15:00") - self._period_return(x, "09:30", "11:30"))
        return values.rename(self.factor_name).reset_index()
