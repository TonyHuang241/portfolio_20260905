from factors.construction.middle_freq_factor_calculation._base_middle_freq_factor import _BaseMiddleFreqFactor


class Mom1y(_BaseMiddleFreqFactor):
    factor_name = "Mom1y"
    lookback_days = 250
    recent_days = 20
    factor_type = "momentum"
    calculation_logic = "从 250 个交易日前到 20 个交易日前的收盘价累计涨幅，即跳过最近一个月的一年动量（Jegadeesh-Titman 1993 的 12-1 动量），使用输入价格的复权口径；任一端点价格缺失或非正时为空。"

    def calculate(self):
        daily = self.daily[["code", "date", "close"]].copy()
        daily["close"] = daily["close"].where(daily["close"].gt(0))

        # 按全市场交易日定位一年前和一个月前的窗口起点，避免个股缺失日线时变成前 N 条记录。
        daily["trade_day"] = daily["date"].rank(method="dense").astype(int)
        daily["year_ago_trade_day"] = daily["trade_day"] - self.lookback_days
        daily["month_ago_trade_day"] = daily["trade_day"] - self.recent_days

        # 按股票代码和窗口起点查询历史收盘价。
        daily = daily.join(daily.set_index(["code", "trade_day"])["close"].rename("year_ago_close"), on=["code", "year_ago_trade_day"])
        daily = daily.join(daily.set_index(["code", "trade_day"])["close"].rename("month_ago_close"), on=["code", "month_ago_trade_day"])

        daily[self.factor_name] = daily["month_ago_close"] / daily["year_ago_close"] - 1
        return daily[["code", "date", self.factor_name]]
