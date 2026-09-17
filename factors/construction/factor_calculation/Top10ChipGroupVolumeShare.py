import pandas as pd

from factors.construction.factor_calculation._base_high_freq_factor import _BaseHighFreqFactor


class Top10ChipGroupVolumeShare(_BaseHighFreqFactor):
    factor_name = "Top10ChipGroupVolumeShare"
    factor_type = "chip_distribution"
    report_factor_code = "doc_vol10_ratio"
    calculation_logic = "按分钟收益率分组后成交量最大的10组占全天成交量比例。"

    def calculate(self):
        minutes = self.minutes.copy()
        minutes["trade_time"] = pd.to_datetime(minutes["trade_time"])
        minutes = minutes.sort_values(["code", "date", "trade_time"])
        def value(group):
            returns = pd.to_numeric(group["close"], errors="coerce").pct_change(fill_method=None)
            returns.iloc[0] = group["close"].iloc[0] / group["open"].iloc[0] - 1
            data = pd.DataFrame({"return": returns, "volume": pd.to_numeric(group["volume"], errors="coerce")}).dropna()
            grouped_volume = data.groupby("return")["volume"].sum()
            return grouped_volume.nlargest(10).sum() / grouped_volume.sum() if grouped_volume.sum() else float("nan")
        return minutes.groupby(["code", "date"]).apply(value).rename(self.factor_name).reset_index()
