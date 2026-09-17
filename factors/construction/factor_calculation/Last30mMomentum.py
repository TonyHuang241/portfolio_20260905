import pandas as pd

from factors.construction.factor_calculation._base_high_freq_factor import _BaseHighFreqFactor


class Last30mMomentum(_BaseHighFreqFactor):
    factor_name = "Last30mMomentum"
    factor_type = "momentum_reversal"
    report_factor_code = "mmt_last30"
    calculation_logic = "14:30至15:00首根开盘价到末根收盘价的收益率；若期间触及涨跌停则置空。"

    def __init__(self, minutes, limit_price=None, **kwargs):
        super().__init__(minutes)
        self.limit_price = limit_price

    def calculate(self):
        minutes = self.minutes.copy()
        minutes["trade_time"] = pd.to_datetime(minutes["trade_time"])
        start, end = pd.Timestamp("14:30").time(), pd.Timestamp("15:00").time()
        minutes = minutes.loc[minutes["trade_time"].dt.time.between(start, end)]
        result = minutes.sort_values(["code", "date", "trade_time"]).groupby(["code", "date"], as_index=False).agg(start=("open", "first"), end=("close", "last"))
        result[self.factor_name] = result["end"] / result["start"] - 1
        if self.limit_price is not None:
            bars = minutes.merge(self.limit_price[["code", "date", "high_limit", "low_limit"]], on=["code", "date"], how="left", validate="many_to_one")
            hit = (bars["close"].eq(bars["high_limit"]) | bars["close"].eq(bars["low_limit"])).groupby([bars["code"], bars["date"]]).any().rename("hit_limit").reset_index()
            result = result.merge(hit, on=["code", "date"], how="left")
            result.loc[result["hit_limit"].fillna(False), self.factor_name] = float("nan")
        return result[["code", "date", self.factor_name]]
