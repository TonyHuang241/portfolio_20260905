import os
from importlib import import_module
from multiprocessing import Pool

import pandas as pd
from tqdm import tqdm


class MiddleFreqFactorConstructor:
    def __init__(self, config):
        self.stock_daily_dir = config["stock_daily_dir"]
        self.backtest_data_path = os.path.join(self.stock_daily_dir, "daily_stock_data_10am.parquet")
        self.stock_st_status_dir = config["stock_st_status_dir"]
        self.financial_data_path = os.path.join(config["fundamentals_dir"], "financial_data.parquet")
        self.output_dir = config["output_dir"]
        self.middle_factor_calc_dir = config["middle_factor_calc_dir"]
        self.start_date = config["start_date"]
        self.end_date = config["end_date"]
        self.update_all = config["update_all"]
        self.processes = config["processes"]
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
        """生成每日股票池：非 ST、连续交易至少 250 天；不区分板块，非主板在评估时剔除。"""
        # 读取时先按时间区间和数值条件下推过滤，只对剩余行做日期字符串转换。
        stock_pool = pd.read_parquet(
            self.backtest_data_path,
            columns=["code", "trade_time"],
            filters=[
                ("trade_time", ">=", pd.Timestamp(min(trade_dates))),
                ("trade_time", "<", pd.Timestamp(max(trade_dates)) + pd.Timedelta(days=1)),
                ("consecutive_trading_days", ">=", 250),
            ],
        )
        # 只对唯一交易日做一次字符串格式化，再按编码映射回每一行。
        day_codes, days = pd.factorize(pd.to_datetime(stock_pool["trade_time"]).dt.normalize())
        stock_pool["date"] = days.strftime("%Y%m%d").take(day_codes)
        stock_pool = stock_pool.loc[stock_pool["date"].isin(trade_dates), ["date", "code"]]

        stock_st_status = pd.read_parquet(os.path.join(self.stock_st_status_dir, "stock_st_status.parquet"), columns=["code", "date", "is_st"])
        stock_pool = stock_pool.merge(stock_st_status, on=["code", "date"], how="left")
        return stock_pool.loc[~stock_pool["is_st"].eq(True), ["date", "code"]]

    @staticmethod
    def _calculate_factor(task):
        """在工作进程中计算单个因子及其滚动均值，筛选股票池后与历史数据合并保存。"""
        factor_name, factor_class, daily, financial_data, stock_pool, history_factor, pending_dates, rolling_windows, output_dir = task
        factor = factor_class(daily=daily, financial_data=financial_data).calculate()
        factor = factor.sort_values(["code", "date"]).reset_index(drop=True)
        factor_by_code = factor.groupby("code")[factor_name]
        for window in rolling_windows:
            factor[f"{factor_name}_{window}_m"] = factor_by_code.rolling(window).mean().droplevel(0).reindex(factor.index)

        factor = factor.loc[factor["date"].isin(pending_dates)]
        factor = factor.merge(stock_pool, on=["code", "date"], how="inner")
        factor = pd.concat([history_factor, factor], ignore_index=True)
        factor = factor.drop_duplicates(["code", "date"], keep="last")
        factor = factor.sort_values(["code", "date"]).reset_index(drop=True)

        factor.to_parquet(os.path.join(output_dir, f"{factor_name}.parquet"), index=False)
        return factor_name

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
        # 计算前剔除开盘价为 0、全天成交额为 0（停牌）的日线，以及 ST 和连续交易不满 250 天的观测；一字日和全天价格恒定但有成交的样本保留。
        daily = daily.loc[daily["date"].ge(history_start)]
        daily = daily.loc[daily["open"].ne(0) & daily["money"].ne(0)]
        stock_pool = self._select_stock_codes(trade_dates[trade_dates.index(history_start):])
        daily = daily.merge(stock_pool, on=["date", "code"], how="inner")
        # 财务数据不按日线回看期截断，保留全部历史公告，供需要同比等多期数据的因子使用。
        financial_data = pd.read_parquet(self.financial_data_path)
        os.makedirs(self.output_dir, exist_ok=True)

        # 按因子并行，每个工作进程完成单个因子的计算、滚动均值和合并保存。
        tasks = [
            (factor_name, factor_class, daily, financial_data, stock_pool, factor_data[factor_name], pending_by_factor[factor_name], rolling_windows, self.output_dir)
            for factor_name, factor_class in self.factors.items() if pending_by_factor[factor_name]
        ]
        with Pool(processes=self.processes) as pool:
            results = pool.imap_unordered(MiddleFreqFactorConstructor._calculate_factor, tasks)
            factor_progress = tqdm(results, total=len(tasks), desc="Updating middle-frequency factors", unit="factor")
            for factor_name in factor_progress:
                factor_progress.set_postfix_str(f"Completed factor: {factor_name}")
