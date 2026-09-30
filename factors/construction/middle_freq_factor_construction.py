import os
from importlib import import_module

import pandas as pd
from tqdm import tqdm


class MiddleFreqFactorConstructor:
    def __init__(self, config):
        self.stock_daily_dir = config["stock_daily_dir"]
        self.backtest_data_path = os.path.join(self.stock_daily_dir, "daily_stock_data_10am.parquet")
        self.stock_st_status_dir = config["stock_st_status_dir"]
        self.output_dir = config["output_dir"]
        self.middle_factor_calc_dir = config["middle_factor_calc_dir"]
        self.start_date = config["start_date"]
        self.end_date = config["end_date"]
        self.update_all = config["update_all"]
        self._regist_factors(config.get("middle_factor_list", []))

    def _regist_factors(self, factor_list):
        """注册中频因子类；列表为空时注册目录中全部因子。"""
        if not factor_list:
            factor_list = [
                file[:-3] for file in sorted(os.listdir(self.middle_factor_calc_dir))
                if not file.startswith("_") and file.endswith(".py") and os.path.isfile(os.path.join(self.middle_factor_calc_dir, file))
            ]
        self.factors = {}
        for factor_name in factor_list:
            module = import_module(f"factors.construction.middle_freq_factor_calculation.{factor_name}")
            self.factors[factor_name] = getattr(module, factor_name)

    def _select_stock_codes(self, trade_dates):
        """生成每日股票池：主板、非 ST、连续交易至少 250 天、全天收盘价非恒定。"""
        stock_pool = pd.read_parquet(self.backtest_data_path, columns=["code", "trade_time", "consecutive_trading_days", "is_constant_close"])
        stock_pool = stock_pool.loc[stock_pool["code"].str.startswith(("000", "001", "002", "003", "600", "601", "603", "605"))]
        stock_pool["date"] = pd.to_datetime(stock_pool["trade_time"]).dt.strftime("%Y%m%d")
        stock_pool = stock_pool.loc[stock_pool["is_constant_close"].eq(False)]
        stock_pool = stock_pool.loc[stock_pool["date"].isin(trade_dates) & stock_pool["consecutive_trading_days"].ge(250), ["date", "code"]]

        stock_st_status = pd.read_parquet(os.path.join(self.stock_st_status_dir, "stock_st_status.parquet"), columns=["code", "date", "is_st"])
        stock_pool = stock_pool.merge(stock_st_status, on=["code", "date"], how="left")
        return stock_pool.loc[~stock_pool["is_st"].eq(True), ["date", "code"]]

    def update_factors(self):
        """一次读取日线，预留因子回看期及滚动窗口，计算完均值后筛选输出。"""
        if not self.factors:
            return

        daily = pd.read_parquet(os.path.join(self.stock_daily_dir, "daily_stock_data.parquet"))
        daily = daily.sort_values(["code", "date"])
        trade_dates = sorted(daily["date"].unique())
        update_dates = {date for date in trade_dates if self.start_date <= date <= self.end_date}
        factor_data = {}
        pending_by_factor = {}
        rolling_windows = (5, 20, 60, 120, 250)

        for factor_name in self.factors:
            factor_path = os.path.join(self.output_dir, f"{factor_name}.parquet")
            legacy_factor_path = os.path.join(self.output_dir, f"{factor_name}.csv")
            if self.update_all == 0 and os.path.exists(factor_path):
                factor_data[factor_name] = pd.read_parquet(factor_path)
                factor_data[factor_name][["code", "date"]] = factor_data[factor_name][["code", "date"]].astype(str)
            elif self.update_all == 0 and os.path.exists(legacy_factor_path):
                factor_data[factor_name] = pd.read_csv(legacy_factor_path, dtype={"code": str, "date": str})
            else:
                factor_data[factor_name] = pd.DataFrame(columns=["code", "date", factor_name])
            pending_by_factor[factor_name] = update_dates - set(factor_data[factor_name]["date"])

        pending_dates = sorted(set.union(*pending_by_factor.values()))
        if not pending_dates:
            return

        history_days = 252 + max(factor.lookback_days for factor in self.factors.values())
        history_start = trade_dates[max(0, trade_dates.index(pending_dates[0]) - history_days)]
        # 输入保留所有股票的完整历史，直至数据最新日期；股票池只筛选输出。
        daily = daily.loc[daily["date"].ge(history_start)]
        stock_pool = self._select_stock_codes(pending_dates)
        os.makedirs(self.output_dir, exist_ok=True)

        for factor_name, factor_class in tqdm(self.factors.items(), desc="Updating middle-frequency factors", unit="factor"):
            if not pending_by_factor[factor_name]:
                continue

            factor = factor_class(daily=daily).calculate()
            factor = factor.sort_values(["code", "date"]).reset_index(drop=True)
            factor_by_code = factor.groupby("code")[factor_name]
            for window in rolling_windows:
                factor[f"{factor_name}_{window}_m"] = factor_by_code.rolling(window).mean().droplevel(0).reindex(factor.index)

            factor = factor.loc[factor["date"].isin(pending_by_factor[factor_name])]
            factor = factor.merge(stock_pool, on=["code", "date"], how="inner")
            factor = pd.concat([factor_data[factor_name], factor], ignore_index=True)
            factor = factor.drop_duplicates(["code", "date"], keep="last")
            factor = factor.sort_values(["code", "date"]).reset_index(drop=True)

            factor.to_parquet(os.path.join(self.output_dir, f"{factor_name}.parquet"), index=False)
