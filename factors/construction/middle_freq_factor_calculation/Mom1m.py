from factors.construction.middle_freq_factor_calculation._base_middle_freq_factor import _BaseMiddleFreqFactor


class Mom1m(_BaseMiddleFreqFactor):
    factor_name = "Mom1m"
    lookback_days = 20
    factor_type = "momentum"
    calculation_logic = "过去 20 个交易日收盘价累计涨幅，使用输入价格的复权口径。"

    def calculate(self):
        daily = self.daily[["code", "date", "close"]].copy()
        daily["close"] = daily["close"].where(daily["close"].gt(0))

        # 按全市场交易日定位窗口起点，避免个股缺失日线时变成前 20 条记录。
        daily["trade_day"] = daily["date"].rank(method="dense").astype(int)
        daily["previous_trade_day"] = daily["trade_day"] - self.lookback_days

        # 按股票代码和窗口起点查询历史收盘价。
        daily = daily.join(
            daily.set_index(["code", "trade_day"])["close"].rename("previous_close"),
            on=["code", "previous_trade_day"],
        )

        daily[self.factor_name] = daily["close"] / daily["previous_close"] - 1
        return daily[["code", "date", self.factor_name]]
