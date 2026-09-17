import numpy as np
import pandas as pd

from factors.construction.factor_calculation._base_high_freq_factor import _BaseHighFreqFactor


class VolumeWeightedReturnCentral90Span(_BaseHighFreqFactor):
    factor_name = "VolumeWeightedReturnCentral90Span"
    factor_type = "chip_distribution"
    report_factor_code = "doc_vol_pdf90bi"
    calculation_logic = "按成交量加权的分钟收益率分布95%分位减5%分位。"

    @staticmethod
    def _weighted_quantile(data, quantile):
        total_weight = data["weight"].sum()
        if data.empty or not np.isfinite(total_weight) or total_weight <= 0:
            return float("nan")
        weights = data["weight"] / total_weight
        position = min(np.searchsorted(weights.cumsum().to_numpy(), quantile, side="left"), len(data) - 1)
        return data["return"].iloc[position]

    def calculate(self):
        minutes = self.minutes.copy()
        minutes["trade_time"] = pd.to_datetime(minutes["trade_time"])
        minutes = minutes.sort_values(["code", "date", "trade_time"])
        def value(group):
            returns = pd.to_numeric(group["close"], errors="coerce").pct_change(fill_method=None)
            returns.iloc[0] = group["close"].iloc[0] / group["open"].iloc[0] - 1
            data = pd.DataFrame({"return": returns, "weight": pd.to_numeric(group["volume"], errors="coerce")}).dropna()
            data = data.loc[data["weight"] > 0].groupby("return", as_index=False)["weight"].sum().sort_values("return")
            return self._weighted_quantile(data, 0.95) - self._weighted_quantile(data, 0.05)
        return minutes.groupby(["code", "date"]).apply(value).rename(self.factor_name).reset_index()
