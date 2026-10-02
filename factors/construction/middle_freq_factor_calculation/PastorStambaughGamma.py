import numpy as np

from factors.construction.middle_freq_factor_calculation._base_middle_freq_factor import _BaseMiddleFreqFactor


class PastorStambaughGamma(_BaseMiddleFreqFactor):
    factor_name = "PastorStambaughGamma"
    lookback_days = 60
    min_observations = 40
    factor_type = "liquidity"
    calculation_logic = "按股票取最近 60 条日线，以当日超额收益对前一交易日收益、前一交易日带符号换手率做带截距的回归，取带符号换手率的系数（Pastor-Stambaugh 2003，成交额改为换手率）。超额收益为个股收益减前一日总市值加权的市场收益，符号取前一日超额收益的符号，换手率为成交额除以总市值；有效样本少于 40 条时为空。系数越负，成交引起的收益反转越强，流动性越差。"

    def calculate(self):
        daily = self.daily[["code", "date", "close", "money", "mkcap"]].copy()
        daily["close"] = daily["close"].where(daily["close"].gt(0))
        daily["mkcap"] = daily["mkcap"].where(daily["mkcap"].gt(0))
        daily["turnover"] = daily["money"] / daily["mkcap"]

        # 按全市场交易日定位前一交易日，个股前一日缺失日线时收益为空，避免跨多日计算收益。
        daily["trade_day"] = daily["date"].rank(method="dense").astype(int)
        daily["previous_trade_day"] = daily["trade_day"] - 1
        daily = daily.join(daily.set_index(["code", "trade_day"])[["close", "mkcap"]].add_prefix("previous_"), on=["code", "previous_trade_day"])
        daily["return"] = daily["close"] / daily["previous_close"] - 1

        # 市场收益以前一日总市值加权，只使用当日收益和前一日市值都有效的股票。
        daily["previous_mkcap"] = daily["previous_mkcap"].where(daily["return"].notna())
        daily["weighted_return"] = daily["return"] * daily["previous_mkcap"]
        daily["market_return"] = daily.groupby("date")["weighted_return"].transform("sum") / daily.groupby("date")["previous_mkcap"].transform("sum")
        daily["excess_return"] = daily["return"] - daily["market_return"]
        daily["signed_turnover"] = np.sign(daily["excess_return"]) * daily["turnover"]

        # 自变量取前一交易日的收益和带符号换手率，因变量为当日超额收益，回归不使用当日之后的数据。
        daily = daily.join(daily.set_index(["code", "trade_day"])[["return", "signed_turnover"]].add_prefix("previous_"), on=["code", "previous_trade_day"])

        # 三个变量任一缺失的行不参与回归：计数记 0，变量置 0，使所有累加和基于同一批样本。
        daily["observation"] = daily[["excess_return", "previous_return", "previous_signed_turnover"]].notna().all(axis=1).astype(int)
        for column in ["excess_return", "previous_return", "previous_signed_turnover"]:
            daily[column] = daily[column].where(daily["observation"].eq(1), 0)
        daily["return_square"] = daily["previous_return"] ** 2
        daily["turnover_square"] = daily["previous_signed_turnover"] ** 2
        daily["return_turnover"] = daily["previous_return"] * daily["previous_signed_turnover"]
        daily["return_excess"] = daily["previous_return"] * daily["excess_return"]
        daily["turnover_excess"] = daily["previous_signed_turnover"] * daily["excess_return"]

        # 按股票对最近 lookback_days 条日线滚动求和，原地覆盖各列。
        daily = daily.sort_values(["code", "date"]).reset_index(drop=True)
        for column in ["observation", "excess_return", "previous_return", "previous_signed_turnover", "return_square", "turnover_square", "return_turnover", "return_excess", "turnover_excess"]:
            daily[column] = daily.groupby("code")[column].rolling(self.lookback_days, min_periods=1).sum().droplevel(0)

        # 由累加和得到去均值后的离差平方和与交叉乘积和，即带截距回归的正规方程。
        daily["observation"] = daily["observation"].where(daily["observation"].ge(self.min_observations))
        daily["return_square"] = daily["return_square"] - daily["previous_return"] ** 2 / daily["observation"]
        daily["turnover_square"] = daily["turnover_square"] - daily["previous_signed_turnover"] ** 2 / daily["observation"]
        daily["return_turnover"] = daily["return_turnover"] - daily["previous_return"] * daily["previous_signed_turnover"] / daily["observation"]
        daily["return_excess"] = daily["return_excess"] - daily["previous_return"] * daily["excess_return"] / daily["observation"]
        daily["turnover_excess"] = daily["turnover_excess"] - daily["previous_signed_turnover"] * daily["excess_return"] / daily["observation"]

        # 两个自变量的回归中，带符号换手率的系数 = (S11 × S2y - S12 × S1y) / (S11 × S22 - S12²)。
        daily["determinant"] = daily["return_square"] * daily["turnover_square"] - daily["return_turnover"] ** 2
        daily[self.factor_name] = (daily["return_square"] * daily["turnover_excess"] - daily["return_turnover"] * daily["return_excess"]) / daily["determinant"].where(daily["determinant"].gt(0))
        return daily[["code", "date", self.factor_name]]
