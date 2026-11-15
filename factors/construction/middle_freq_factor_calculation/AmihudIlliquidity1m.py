from factors.construction.middle_freq_factor_calculation._base_middle_freq_factor import _BaseMiddleFreqFactor


class AmihudIlliquidity1m(_BaseMiddleFreqFactor):
    factor_name = "AmihudIlliquidity1m"
    lookback_days = 20
    factor_type = "liquidity"
    calculation_logic = "最近 20 个交易日 Amihud 非流动性的均值，每日 Amihud 非流动性为日收益率绝对值除以当日成交额，收益使用输入价格的复权口径；窗口内任一交易日的值为空（缺失日线、不在样本内或前一交易日缺失日线）时为空。"

    def calculate(self):
        daily = self.daily[["code", "date", "close", "money"]].copy()
        daily["close"] = daily["close"].where(daily["close"].gt(0))
        daily["money"] = daily["money"].where(daily["money"].gt(0))

        # 按全市场交易日定位前一交易日，个股前一日缺失日线时收益为空，避免跨多日计算收益。
        daily["trade_day"] = daily["date"].rank(method="dense").astype(int)
        daily["previous_trade_day"] = daily["trade_day"] - 1
        daily = daily.join(daily.set_index(["code", "trade_day"])["close"].rename("previous_close"), on=["code", "previous_trade_day"])
        daily["amihud"] = (daily["close"] / daily["previous_close"] - 1).abs() / daily["money"]

        # 按股票对最近 lookback_days 条日线求均值；构建器已为每只股票补齐全部交易日，窗口内任一天为空则结果为空。
        daily = daily.sort_values(["code", "date"]).reset_index(drop=True)
        daily[self.factor_name] = daily.groupby("code")["amihud"].rolling(self.lookback_days).mean().droplevel(0)
        return daily[["code", "date", self.factor_name]]
