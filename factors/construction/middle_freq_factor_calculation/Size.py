import numpy as np

from factors.construction.middle_freq_factor_calculation._base_middle_freq_factor import _BaseMiddleFreqFactor


class Size(_BaseMiddleFreqFactor):
    factor_name = "Size"
    lookback_days = 0
    factor_type = "size"
    calculation_logic = "当日总市值的自然对数，市值缺失或非正时为空；取对数以压缩市值的右偏分布（Fama-French 1993 的 ME）。"

    def calculate(self):
        daily = self.daily[["code", "date", "mkcap"]].copy()
        daily[self.factor_name] = np.log(daily["mkcap"].where(daily["mkcap"].gt(0)))
        return daily[["code", "date", self.factor_name]]
