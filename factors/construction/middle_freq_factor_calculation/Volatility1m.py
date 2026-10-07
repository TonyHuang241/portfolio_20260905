from factors.construction.middle_freq_factor_calculation._base_middle_freq_factor import _BaseMiddleFreqFactor


class Volatility1m(_BaseMiddleFreqFactor):
    factor_name = "Volatility1m"
    lookback_days = 20
    min_observations = 15
    factor_type = "volatility"
    calculation_logic = "按股票取最近 20 条日线，计算日收益率的样本标准差（总波动率），收益使用输入价格的复权口径；个股前一交易日缺失日线时当日收益为空，有效样本少于 15 条时为空。"

    def calculate(self):
        daily = self.daily[["code", "date", "close"]].copy()
        daily["close"] = daily["close"].where(daily["close"].gt(0))

        # 按全市场交易日定位前一交易日，个股前一日缺失日线时收益为空，避免跨多日计算收益。
        daily["trade_day"] = daily["date"].rank(method="dense").astype(int)
        daily["previous_trade_day"] = daily["trade_day"] - 1
        daily = daily.join(daily.set_index(["code", "trade_day"])["close"].rename("previous_close"), on=["code", "previous_trade_day"])
        daily["return"] = daily["close"] / daily["previous_close"] - 1

        # 按股票对最近 lookback_days 条日线滚动计算标准差，空收益不计入有效样本。
        daily = daily.sort_values(["code", "date"]).reset_index(drop=True)
        daily[self.factor_name] = daily.groupby("code")["return"].rolling(self.lookback_days, min_periods=self.min_observations).std().droplevel(0)
        return daily[["code", "date", self.factor_name]]
