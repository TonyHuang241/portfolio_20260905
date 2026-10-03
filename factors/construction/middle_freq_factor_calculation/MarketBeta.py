from factors.construction.middle_freq_factor_calculation._base_middle_freq_factor import _BaseMiddleFreqFactor


class MarketBeta(_BaseMiddleFreqFactor):
    factor_name = "MarketBeta"
    lookback_days = 250
    min_observations = 120
    factor_type = "beta"
    calculation_logic = "按股票取最近 250 条日线，以个股日收益对市场日收益做带截距的回归，取市场收益的系数（CAPM 市场 beta）。市场收益为全部股票按前一日总市值加权的日收益，收益使用输入价格的复权口径；有效样本少于 120 条时为空。"

    def calculate(self):
        daily = self.daily[["code", "date", "close", "mkcap"]].copy()
        daily["close"] = daily["close"].where(daily["close"].gt(0))
        daily["mkcap"] = daily["mkcap"].where(daily["mkcap"].gt(0))

        # 按全市场交易日定位前一交易日，个股前一日缺失日线时收益为空，避免跨多日计算收益。
        daily["trade_day"] = daily["date"].rank(method="dense").astype(int)
        daily["previous_trade_day"] = daily["trade_day"] - 1
        daily = daily.join(daily.set_index(["code", "trade_day"])[["close", "mkcap"]].add_prefix("previous_"), on=["code", "previous_trade_day"])
        daily["return"] = daily["close"] / daily["previous_close"] - 1

        # 市场收益以前一日总市值加权，只使用当日收益和前一日市值都有效的股票。
        daily["previous_mkcap"] = daily["previous_mkcap"].where(daily["return"].notna())
        daily["weighted_return"] = daily["return"] * daily["previous_mkcap"]
        daily["market_return"] = daily.groupby("date")["weighted_return"].transform("sum") / daily.groupby("date")["previous_mkcap"].transform("sum")

        # 个股收益或市场收益缺失的行不参与回归：计数记 0，变量置 0，使所有累加和基于同一批样本。
        daily["observation"] = daily[["return", "market_return"]].notna().all(axis=1).astype(int)
        for column in ["return", "market_return"]:
            daily[column] = daily[column].where(daily["observation"].eq(1), 0)
        daily["market_square"] = daily["market_return"] ** 2
        daily["return_market"] = daily["return"] * daily["market_return"]

        # 按股票对最近 lookback_days 条日线滚动求和，原地覆盖各列。
        daily = daily.sort_values(["code", "date"]).reset_index(drop=True)
        for column in ["observation", "return", "market_return", "market_square", "return_market"]:
            daily[column] = daily.groupby("code")[column].rolling(self.lookback_days, min_periods=1).sum().droplevel(0)

        # 单变量带截距回归的斜率 = 去均值后的交叉乘积和 / 市场收益的离差平方和。
        daily["observation"] = daily["observation"].where(daily["observation"].ge(self.min_observations))
        daily["market_square"] = daily["market_square"] - daily["market_return"] ** 2 / daily["observation"]
        daily["return_market"] = daily["return_market"] - daily["return"] * daily["market_return"] / daily["observation"]
        daily[self.factor_name] = daily["return_market"] / daily["market_square"].where(daily["market_square"].gt(0))
        return daily[["code", "date", self.factor_name]]
