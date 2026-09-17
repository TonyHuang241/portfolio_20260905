import numpy as np
import pandas as pd

from factors.construction.factor_calculation._base_high_freq_factor import _BaseHighFreqFactor


class VolumeWeightedReturnKurtosis(_BaseHighFreqFactor):
    factor_name = "VolumeWeightedReturnKurtosis"
    factor_type = "chip_distribution"
    report_factor_code = "doc_kurt"
    calculation_logic = "按分钟收益率分组并以成交量为权重，计算收益率分布峰度。"

    @staticmethod
    def _distribution(group):
        returns = pd.to_numeric(group["close"], errors="coerce").pct_change(fill_method=None)
        returns.iloc[0] = group["close"].iloc[0] / group["open"].iloc[0] - 1
        data = pd.DataFrame({"return": returns, "weight": pd.to_numeric(group["volume"], errors="coerce")}).dropna()
        data = data.loc[data["weight"] > 0].groupby("return", as_index=False)["weight"].sum()
        total_weight = data["weight"].sum()
        if data.empty or not np.isfinite(total_weight) or total_weight <= 0:
            return data.iloc[0:0]
        data["weight"] /= total_weight
        return data

    def calculate(self):
        minutes = self.minutes.copy()
        minutes["trade_time"] = pd.to_datetime(minutes["trade_time"])
        minutes = minutes.sort_values(["code", "date", "trade_time"])
        def value(group):
            data = self._distribution(group)
            if data.empty:
                return float("nan")
            mean = (data["return"] * data["weight"]).sum()
            variance = (np.square(data["return"] - mean) * data["weight"]).sum()
            std = np.sqrt(variance)
            if "kurt" == "std":
                return std
            order = 3 if "kurt" == "skew" else 4
            return (np.power((data["return"] - mean) / std, order) * data["weight"]).sum() if std else float("nan")
        return minutes.groupby(["code", "date"]).apply(value).rename(self.factor_name).reset_index()
