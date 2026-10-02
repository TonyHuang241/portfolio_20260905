from factors.construction.middle_freq_factor_calculation._base_middle_freq_factor import _BaseMiddleFreqFactor


class HighLowAmihud(_BaseMiddleFreqFactor):
    factor_name = "HighLowAmihud"
    lookback_days = 0
    factor_type = "liquidity"
    calculation_logic = "当日最高价与最低价之差除以收盘价，再除以当日成交额；以日内振幅替代收益率绝对值，不受隔夜跳空影响；一字日振幅为 0，结果为 0。"

    def calculate(self):
        daily = self.daily[["code", "date", "high", "low", "close", "money"]].copy()
        daily["low"] = daily["low"].where(daily["low"].gt(0))
        daily["close"] = daily["close"].where(daily["close"].gt(0))
        daily["money"] = daily["money"].where(daily["money"].gt(0))

        daily[self.factor_name] = (daily["high"] - daily["low"]) / daily["close"] / daily["money"]
        return daily[["code", "date", self.factor_name]]
