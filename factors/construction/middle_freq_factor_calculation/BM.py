import pandas as pd

from factors.construction.middle_freq_factor_calculation._base_middle_freq_factor import _BaseMiddleFreqFactor


class BM(_BaseMiddleFreqFactor):
    factor_name = "BM"
    lookback_days = 0
    factor_type = "value"
    calculation_logic = "最近一次公告财报的账面净资产除以当日总市值（Fama-French 1993 的 B/M）；账面净资产为总资产减总负债，含少数股东权益；按公告日向前匹配，只使用当日及以前已公告的财报；净资产或市值非正时为空。"

    def calculate(self):
        daily = self.daily[["code", "date", "mkcap"]].copy()
        financial_data = self.financial_data[["code", "date", "total_assets", "total_liability"]].copy()
        financial_data["book_equity"] = financial_data["total_assets"] - financial_data["total_liability"]

        # 公告日多为非交易日，按日期向前匹配最近一次公告；merge_asof 需要数值型日期键且两表均按其排序。
        daily["date_number"] = daily["date"].astype(int)
        financial_data["date_number"] = financial_data["date"].astype(int)
        daily = pd.merge_asof(daily.sort_values("date_number"), financial_data[["code", "date_number", "book_equity"]].sort_values("date_number"), on="date_number", by="code", direction="backward")

        daily[self.factor_name] = daily["book_equity"].where(daily["book_equity"].gt(0)) / daily["mkcap"].where(daily["mkcap"].gt(0))
        return daily[["code", "date", self.factor_name]]
