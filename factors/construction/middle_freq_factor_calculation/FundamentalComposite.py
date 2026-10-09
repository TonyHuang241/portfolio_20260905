import numpy as np
import pandas as pd

from factors.construction.middle_freq_factor_calculation._base_middle_freq_factor import _BaseMiddleFreqFactor


class FundamentalComposite(_BaseMiddleFreqFactor):
    factor_name = "FundamentalComposite"
    lookback_days = 0
    components = ["dGM_q", "SUE_OP", "SUR", "CFO_TA"]
    factor_type = "fundamental"
    calculation_logic = "四个财报成分的等权合成：单季毛利率同比变化 dGM_q；单季营业利润超预期 SUE_OP（同比差额除以最近 8 个季度同比差额的标准差，至少 4 个）；单季营收超预期 SUR（同法）；TTM 经营现金流除以期初期末平均总资产 CFO_TA。每日在样本内取各成分的截面分位，对有值的成分求平均，全部缺失时为空。财报按公告日向前匹配，同比等用到的较早报告期须已公告，同日公告多期取最新报告期。"

    def calculate(self):
        daily = self.daily[["code", "date", "close"]].copy()
        financial_data = self.financial_data[["code", "date", "year", "quarter", "operating_revenue_q", "operating_cost_q", "operating_profit_q", "net_operate_cash_flow_ttm", "total_assets"]].copy()

        # 每只股票补齐全部报告期，缺失的报告期各列为空，使按条 shift(4) 为上年同期、rolling(8) 为最近 8 个季度。
        financial_data["quarter_number"] = financial_data["year"] * 4 + financial_data["quarter"] - 1
        financial_data = financial_data.set_index(["code", "quarter_number"]).reindex(pd.MultiIndex.from_product([sorted(financial_data["code"].unique()), range(financial_data["quarter_number"].min(), financial_data["quarter_number"].max() + 1)], names=["code", "quarter_number"])).reset_index()

        financial_data["gross_margin_q"] = (financial_data["operating_revenue_q"] - financial_data["operating_cost_q"]) / financial_data["operating_revenue_q"].where(financial_data["operating_revenue_q"].gt(0))
        financial_data["dGM_q"] = financial_data["gross_margin_q"] - financial_data.groupby("code")["gross_margin_q"].shift(4)
        for component, column in (("SUE_OP", "operating_profit_q"), ("SUR", "operating_revenue_q")):
            financial_data[f"{column}_change"] = financial_data[column] - financial_data.groupby("code")[column].shift(4)
            financial_data[component] = financial_data[f"{column}_change"] / financial_data.groupby("code")[f"{column}_change"].rolling(8, min_periods=4).std().droplevel(0)
        financial_data["total_assets"] = financial_data["total_assets"].where(financial_data["total_assets"].gt(0))
        financial_data["CFO_TA"] = financial_data["net_operate_cash_flow_ttm"] / ((financial_data["total_assets"] + financial_data.groupby("code")["total_assets"].shift(4)) / 2)
        financial_data[self.components] = financial_data[self.components].replace([np.inf, -np.inf], np.nan)

        # 同比和标准差用到较早的报告期，可得日取本期及之前各期公告日的累计最大值；补齐的空报告期不参与匹配。
        financial_data["date_number"] = financial_data["date"].astype(float)
        financial_data["date_number"] = financial_data.groupby("code")["date_number"].cummax()
        financial_data = financial_data.loc[financial_data["date"].notna()]
        financial_data["date_number"] = financial_data["date_number"].astype(int)

        # 按日期向前匹配最近一次可得的报告期，同日可得多期时排在最后的最新报告期被选中；不在样本内（收盘价为空）的观测不参与排序。
        daily["date_number"] = daily["date"].astype(int)
        daily = pd.merge_asof(daily.sort_values("date_number"), financial_data.sort_values(["date_number", "quarter_number"])[["code", "date_number"] + self.components], on="date_number", by="code", direction="backward")
        daily[self.components] = daily[self.components].where(daily["close"].notna(), axis=0)

        # 每日在样本内取各成分的截面分位，对有值的成分等权平均。
        daily[self.factor_name] = daily.groupby("date")[self.components].rank(pct=True).mean(axis=1)
        return daily[["code", "date", self.factor_name]]
