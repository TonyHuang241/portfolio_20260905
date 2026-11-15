from pathlib import Path

import numpy as np
import pandas as pd


class FactorCorrelationAnalyzer:
    """多因子截面分析：计算因子之间的截面相关性矩阵，供因子评估报告使用。"""
    def __init__(self, config, factor_list):
        self.factor_dir = Path(config["factor_data_dir"])
        self.monthly_mkcap_path = Path(config["stock_minutes_dir"]).parent / "fundamentals" / "mkcap_monthly.parquet"
        self.start_date = config["start_date"]
        self.stock_pool = config["stock_pool"]
        self.stock_board = config["stock_board"]
        # 因子文件名即因子列名，每个文件只读与文件名同名的列；矩阵中的因子顺序与 factor_list 一致。
        self.factor_list = factor_list

    def _load_factors(self):
        """读取全部因子，按单因子评估的板块和小市值股票池筛选，再逐日截面标准化。"""
        factor = pd.concat([pd.read_parquet(self.factor_dir / f"{factor_name}.parquet", columns=["code", "date", factor_name], filters=[("date", ">=", self.start_date)]).set_index(["code", "date"]) for factor_name in self.factor_list], axis=1)
        factor = factor.reset_index()
        if self.stock_board == "main board":
            factor = factor.loc[factor["code"].str.startswith(("000", "001", "002", "003", "600", "601", "603", "605"))]

        # 与单因子评估一致：按因子日期所在月份取上月末市值，每日保留市值最小的 stock_pool 只股票。
        monthly_mkcap = pd.read_parquet(self.monthly_mkcap_path, columns=["code", "date", "mkcap"])
        monthly_mkcap = monthly_mkcap.rename(columns={"date": "month"})
        factor["month"] = factor["date"].str[:6] + "01"
        factor = factor.merge(monthly_mkcap, on=["code", "month"])
        factor = factor.dropna(subset=["mkcap"])
        factor = factor.sort_values("mkcap", kind="stable")
        factor = factor.groupby("date", sort=False).head(self.stock_pool)
        factor = factor.sort_values(["date", "code"]).reset_index(drop=True)

        # 无穷值视为缺失；逐日减截面均值、除以截面标准差，标准差为 0 时结果为空。
        factor[self.factor_list] = factor[self.factor_list].replace([np.inf, -np.inf], np.nan)
        factor[self.factor_list] = (factor[self.factor_list] - factor.groupby("date")[self.factor_list].transform("mean")) / factor.groupby("date")[self.factor_list].transform("std").replace(0, np.nan)
        return factor

    def correlation_matrix(self):
        """读取全部因子，逐日计算两两因子的截面 Pearson 相关系数，再对全部日期取均值。"""
        factor = self._load_factors()
        correlation = factor.groupby("date")[self.factor_list].corr()
        return correlation.groupby(level=1).mean().reindex(index=self.factor_list, columns=self.factor_list)
