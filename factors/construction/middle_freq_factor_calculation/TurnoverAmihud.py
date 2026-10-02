from factors.construction.middle_freq_factor_calculation._base_middle_freq_factor import _BaseMiddleFreqFactor


class TurnoverAmihud(_BaseMiddleFreqFactor):
    factor_name = "TurnoverAmihud"
    lookback_days = 1
    factor_type = "liquidity"
    calculation_logic = "日收益率绝对值除以当日换手率，换手率为成交额除以总市值，收益使用输入价格的复权口径；以换手率替代成交额，剔除 Amihud 非流动性的市值暴露（Florackis 2011）。"

    def calculate(self):
        daily = self.daily[["code", "date", "close", "money", "mkcap"]].copy()
        daily["close"] = daily["close"].where(daily["close"].gt(0))
        daily["mkcap"] = daily["mkcap"].where(daily["mkcap"].gt(0))
        daily["turnover"] = daily["money"] / daily["mkcap"]
        daily["turnover"] = daily["turnover"].where(daily["turnover"].gt(0))

        # 按全市场交易日定位前一交易日，个股前一日缺失日线时收益为空，避免跨多日计算收益。
        daily["trade_day"] = daily["date"].rank(method="dense").astype(int)
        daily["previous_trade_day"] = daily["trade_day"] - self.lookback_days
        daily = daily.join(daily.set_index(["code", "trade_day"])["close"].rename("previous_close"), on=["code", "previous_trade_day"])

        daily[self.factor_name] = (daily["close"] / daily["previous_close"] - 1).abs() / daily["turnover"]
        return daily[["code", "date", self.factor_name]]
