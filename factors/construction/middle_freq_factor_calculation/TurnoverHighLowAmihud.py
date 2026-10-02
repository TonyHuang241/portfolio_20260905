from factors.construction.middle_freq_factor_calculation._base_middle_freq_factor import _BaseMiddleFreqFactor


class TurnoverHighLowAmihud(_BaseMiddleFreqFactor):
    factor_name = "TurnoverHighLowAmihud"
    lookback_days = 0
    factor_type = "liquidity"
    calculation_logic = "当日最高价与最低价之差除以收盘价，再除以当日换手率，换手率为成交额除以总市值；以换手率替代成交额，剔除 HighLowAmihud 的市值暴露；一字日振幅为 0，结果为 0。"

    def calculate(self):
        daily = self.daily[["code", "date", "high", "low", "close", "money", "mkcap"]].copy()
        daily["low"] = daily["low"].where(daily["low"].gt(0))
        daily["close"] = daily["close"].where(daily["close"].gt(0))
        daily["mkcap"] = daily["mkcap"].where(daily["mkcap"].gt(0))
        daily["turnover"] = daily["money"] / daily["mkcap"]
        daily["turnover"] = daily["turnover"].where(daily["turnover"].gt(0))

        daily[self.factor_name] = (daily["high"] - daily["low"]) / daily["close"] / daily["turnover"]
        return daily[["code", "date", self.factor_name]]
