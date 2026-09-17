import pandas as pd

from factors.construction.factor_calculation._base_high_freq_factor import _BaseHighFreqFactor


class Last50BarWeightedReturn(_BaseHighFreqFactor):
    factor_name = "Last50BarWeightedReturn"
    factor_type = "trade_activity"
    report_factor_code = "trade_bottom50retRatio"
    calculation_logic = "最后50根分钟K线收益率按其全天成交量占比加权求和。"

    def calculate(self):
        minutes = self.minutes.copy()
        minutes["trade_time"] = pd.to_datetime(minutes["trade_time"])
        minutes = minutes.sort_values(["code", "date", "trade_time"])
        def value(group):
            returns = pd.to_numeric(group["close"], errors="coerce").pct_change(fill_method=None)
            returns.iloc[0] = group["close"].iloc[0] / group["open"].iloc[0] - 1
            selected = group.iloc[:50] if "last" == "first" else group.iloc[-50:]
            selected_returns = returns.loc[selected.index]
            if "all" == "negative":
                selected_returns = selected_returns[selected_returns < 0].abs()
            elif "all" == "positive":
                selected_returns = selected_returns[selected_returns > 0]
            selected_returns = selected_returns.dropna()
            total_volume = pd.to_numeric(group["volume"], errors="coerce").sum(min_count=1)
            if selected_returns.empty or pd.isna(total_volume) or total_volume <= 0:
                return float("nan")
            volume_share = pd.to_numeric(group.loc[selected_returns.index, "volume"], errors="coerce") / total_volume
            weighted_return = selected_returns * volume_share
            return weighted_return.sum()
        return minutes.groupby(["code", "date"]).apply(value).rename(self.factor_name).reset_index()
