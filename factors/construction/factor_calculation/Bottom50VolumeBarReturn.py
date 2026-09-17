import pandas as pd

from factors.construction.factor_calculation._base_high_freq_factor import _BaseHighFreqFactor


class Bottom50VolumeBarReturn(_BaseHighFreqFactor):
    factor_name = "Bottom50VolumeBarReturn"
    factor_type = "momentum_reversal"
    report_factor_code = "mmt_bottom50VolumeRet"
    calculation_logic = "成交量最小的50根分钟K线收益率的复合值。"

    def calculate(self):
        minutes = self.minutes.copy()
        minutes["trade_time"] = pd.to_datetime(minutes["trade_time"])
        minutes = minutes.sort_values(["code", "date", "trade_time"])
        def value(group):
            returns = group["close"].pct_change(fill_method=None)
            returns.iloc[0] = group["close"].iloc[0] / group["open"].iloc[0] - 1
            volume = pd.to_numeric(group["volume"], errors="coerce").dropna()
            selected = volume.nsmallest(50).index
            selected_returns = returns.loc[selected].dropna()
            return (1 + selected_returns).prod() - 1 if not selected_returns.empty else float("nan")
        return minutes.groupby(["code", "date"]).apply(value).rename(self.factor_name).reset_index()
