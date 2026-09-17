import numpy as np
import pandas as pd

from factors.construction.factor_calculation._base_high_freq_factor import _BaseHighFreqFactor


class UpsideVolatilityRatio(_BaseHighFreqFactor):
    factor_name = "UpsideVolatilityRatio"
    factor_type = "volatility"
    report_factor_code = "vol_upRatio"
    calculation_logic = "上行波动率占全部分钟收益率均方根波动率的比例。"

    def calculate(self):
        minutes = self.minutes.copy()
        minutes["trade_time"] = pd.to_datetime(minutes["trade_time"])
        minutes = minutes.sort_values(["code", "date", "trade_time"])
        def value(group):
            returns = pd.to_numeric(group["close"], errors="coerce").pct_change(fill_method=None)
            returns.iloc[0] = group["close"].iloc[0] / group["open"].iloc[0] - 1
            returns = returns.replace([np.inf, -np.inf], np.nan).dropna()
            selected = returns[returns > 0] if "up" == "up" else returns[returns < 0]
            semivolatility = np.sqrt(np.mean(np.square(selected))) if not selected.empty else 0.0
            if "ratio" == "vol":
                return semivolatility
            total = np.sqrt(np.mean(np.square(returns))) if not returns.empty else np.nan
            return semivolatility / total if total else np.nan
        return minutes.groupby(["code", "date"]).apply(value).rename(self.factor_name).reset_index()
