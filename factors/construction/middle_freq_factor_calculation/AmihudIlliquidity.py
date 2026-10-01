from factors.construction.middle_freq_factor_calculation._base_middle_freq_factor import _BaseMiddleFreqFactor


class AmihudIlliquidity(_BaseMiddleFreqFactor):
    factor_name = "AmihudIlliquidity"
    lookback_days = 1
    factor_type = "liquidity"
    calculation_logic = "日收益率绝对值除以当日成交额，收益使用输入价格的复权口径；多日 Amihud 非流动性见构建器输出的滚动均值列。"

    def calculate(self):
        daily = self.daily[["code", "date", "close", "money"]].copy()
        daily["close"] = daily["close"].where(daily["close"].gt(0))
        daily["money"] = daily["money"].where(daily["money"].gt(0))

        # 按全市场交易日定位前一交易日，个股前一日缺失日线时收益为空，避免跨多日计算收益。
        daily["trade_day"] = daily["date"].rank(method="dense").astype(int)
        daily["previous_trade_day"] = daily["trade_day"] - self.lookback_days
        daily = daily.join(daily.set_index(["code", "trade_day"])["close"].rename("previous_close"), on=["code", "previous_trade_day"])

        daily[self.factor_name] = (daily["close"] / daily["previous_close"] - 1).abs() / daily["money"]
        return daily[["code", "date", self.factor_name]]
