import os
from io import BytesIO
from importlib import import_module
from multiprocessing import Pool
from zipfile import ZipFile

import numpy as np
import pandas as pd
from tqdm import tqdm


class HighFreqFactorConstructor:
    def __init__(self, config):
        self.stock_minutes_dir = config["stock_minutes_dir"]
        self.stock_daily_dir = config["stock_daily_dir"]
        self.stock_st_status_dir = config["stock_st_status_dir"]
        self.adj_factor_dir = config["adj_factor_dir"]
        self.output_dir = config["output_dir"]
        self.high_factor_calc_dir = config["high_factor_calc_dir"]

        self.start_date = config["start_date"]
        self.end_date = config["end_date"]
        self.update_all = config["update_all"]
        self.processes = config["processes"]

        self._update_trade_date()
        self._regist_factors(config.get("factor_list", []))

        pass

    def _update_trade_date(self):
        """获取日频数据中存在的全部交易日期。"""
        trade_dates = pd.read_parquet(os.path.join(self.stock_daily_dir, "daily_stock_data.parquet"), columns=["date"])["date"]
        trade_dates = trade_dates.drop_duplicates()
        self.trade_date = sorted(trade_dates)

    def _regist_factors(self, factor_list=None):
        """注册指定因子，列表为空时注册全部因子；文件名、类名与因子名须一致。"""
        if factor_list:
            self.factors = list(factor_list)
            return

        self.factors = [
            file[:-3] for file in sorted(os.listdir(self.high_factor_calc_dir))
            if not file.startswith("_") and file.endswith(".py") and os.path.isfile(os.path.join(self.high_factor_calc_dir, file))
        ]

    @staticmethod
    def _read_minutes_file(stock_minutes_dir, trade_date, stock_codes):
        """只读取指定股票的日度分钟回报数据。"""
        # 分钟文件保留原始后缀，兼容六位股票池代码及沪市历史上的两种后缀。
        stock_codes = np.asarray([f"{code}{suffix}" for code in stock_codes for suffix in ("", ".SH", ".SS", ".SZ", ".BJ")], dtype=str)
        year_path = os.path.join(stock_minutes_dir, trade_date[:4])
        file_name = f"{trade_date}.parquet"
        minutes_path = os.path.join(year_path, file_name)
        if os.path.isfile(minutes_path):
            return pd.read_parquet(minutes_path, engine="pyarrow", filters=[("code", "in", stock_codes)])

        with ZipFile(f"{year_path}.zip") as archive:
            # Parquet 会随机读取文件，先一次性解压，避免 ZIP 文件流反复解压。
            return pd.read_parquet(BytesIO(archive.read(file_name)), engine="pyarrow", filters=[("code", "in", stock_codes)])

    @staticmethod
    def _calculate_trade_date(task):
        """在工作进程中读取、清洗单日数据，并串行计算当天需要更新的因子。"""
        trade_date, stock_minutes_dir, factor_names, stock_codes = task
        minutes = HighFreqFactorConstructor._read_minutes_file(stock_minutes_dir, trade_date, stock_codes)
        minutes["code"] = minutes["code"].astype(str).str.split(".", n=1).str[0].str.zfill(6)
        minutes = minutes.rename(columns={"amount": "money", "vol": "volume"})
        minutes["trade_time"] = pd.to_datetime(minutes["trade_time"])
        minutes["date"] = minutes["trade_time"].dt.normalize()
        # 只格式化不同日期，避免对每日上百万条分钟记录重复转换字符串。
        minutes["date"] = minutes["date"].map({date: date.strftime("%Y%m%d") for date in minutes["date"].dropna().unique()})

        # 全天成交量为 0（停牌，成交量缺失按 0 计），则清空该股票当天全部行情，保留代码和时间。
        invalid_day = minutes.groupby(["code", "date"])["volume"].transform("sum").eq(0)
        minutes.loc[invalid_day, minutes.columns.difference(["code", "trade_time", "date"])] = np.nan

        results = {}
        for factor_name in factor_names:
            module = import_module(f"factors.construction.high_freq_factor_calculation.{factor_name}")
            factor = getattr(module, factor_name)(minutes=minutes)
            results[factor_name] = factor.calculate()
        return trade_date, results

    def _select_stock_codes(self, trade_dates):
        """基于 10 点回测数据生成每日股票池：主板、非 ST、连续交易至少 250 天；连续交易天数沿用数据更新按交易日序号算出的结果，与中频因子一致。"""
        stock_pool = pd.read_parquet(os.path.join(self.stock_daily_dir, "daily_stock_data_10am.parquet"), columns=["code", "trade_time"], filters=[("consecutive_trading_days", ">=", 250)])
        stock_pool = stock_pool.loc[stock_pool["code"].str.startswith(("000", "001", "002", "003", "600", "601", "603", "605"))]
        # 只对唯一交易日做一次字符串格式化，再按编码映射回每一行。
        day_codes, days = pd.factorize(stock_pool["trade_time"].dt.normalize())
        stock_pool["date"] = days.strftime("%Y%m%d").take(day_codes)
        stock_pool = stock_pool.loc[stock_pool["date"].isin(trade_dates), ["date", "code"]]

        stock_st_status = pd.read_parquet(os.path.join(self.stock_st_status_dir, "stock_st_status.parquet"), columns=["code", "date", "is_st"])
        stock_pool = stock_pool.merge(stock_st_status, on=["code", "date"], how="left")
        return stock_pool.loc[~stock_pool["is_st"].eq(True), ["date", "code"]]

    def update_factors(self):
        """逐日更新因子数据。"""
        trade_dates = sorted(date for date in self.trade_date if self.start_date <= date <= self.end_date)
        factor_data = {}
        exist_dates = {}

        # 读取历史数据，避免全量更新
        for factor_name in self.factors:
            factor_path = f"{self.output_dir}/{factor_name}.parquet"
            legacy_factor_path = f"{self.output_dir}/{factor_name}.csv"
            if self.update_all == 0 and os.path.exists(factor_path):
                factor_data[factor_name] = pd.read_parquet(factor_path, columns=["code", "date", factor_name])
                factor_data[factor_name][["code", "date"]] = factor_data[factor_name][["code", "date"]].astype(str)
                exist_dates[factor_name] = set(factor_data[factor_name]["date"].unique())
            elif self.update_all == 0 and os.path.exists(legacy_factor_path):
                factor_data[factor_name] = pd.read_csv(legacy_factor_path, usecols=["code", "date", factor_name], dtype={"code": str, "date": str})
                exist_dates[factor_name] = set(factor_data[factor_name]["date"].unique())
            else:
                factor_data[factor_name] = pd.DataFrame()
                exist_dates[factor_name] = set()
            factor_data[factor_name] = [factor_data[factor_name]]

        common_exist_dates = set.intersection(*exist_dates.values())

        pending_dates = [date for date in trade_dates if date not in common_exist_dates]
        stock_pool = self._select_stock_codes(pending_dates)
        stock_codes_by_date = stock_pool.groupby("date")["code"].agg(list).to_dict()

        # 股票池提前计算完毕；按交易日并行，每个工作进程内串行计算当天需要更新的因子。
        tasks = (
            (
                trade_date,
                self.stock_minutes_dir,
                [name for name in self.factors if trade_date not in exist_dates[name]],
                stock_codes_by_date.get(trade_date, []),
            )
            for trade_date in pending_dates
        )
        with Pool(processes=self.processes) as pool:
            results = pool.imap_unordered(HighFreqFactorConstructor._calculate_trade_date, tasks)
            trade_date_progress = tqdm(results, total=len(pending_dates), desc="Updating raw factor values", unit="trading day")
            for trade_date, daily_results in trade_date_progress:
                trade_date_progress.set_postfix_str(f"Completed date: {trade_date}")
                for factor_name, daily_factor in daily_results.items():
                    factor_data[factor_name].append(daily_factor)

        # 逐个计算并保存，避免同时保留全部因子的滚动指标。
        os.makedirs(self.output_dir, exist_ok=True)
        for factor_name in tqdm(self.factors, desc="Updating derived factor values", unit="factor"):
            factor = pd.concat(factor_data.pop(factor_name), ignore_index=True)
            factor = factor.drop_duplicates(["code", "date"], keep="last")
            factor = factor.sort_values(["code", "date"]).reset_index(drop=True)
            # 每只股票补齐全部交易日，停牌等缺失日期因子值为空、不向前补全，使按条滚动的窗口即为最近 N 个交易日；滚动后去掉补齐的行。
            factor["is_calculated"] = True
            factor = factor.set_index(["code", "date"]).reindex(pd.MultiIndex.from_product([factor["code"].unique(), sorted(factor["date"].unique())], names=["code", "date"])).reset_index()
            factor_by_code = factor.groupby("code")[factor_name]
            for window in (5, 20, 60, 120, 250):
                rolling = factor_by_code.rolling(window)
                factor[f"{factor_name}_{window}_m"] = rolling.mean().droplevel(0).reindex(factor.index)
                # factor[f"{factor_name}_{window}_std"] = rolling.std().droplevel(0).reindex(factor.index)
            factor = factor.loc[factor["is_calculated"].eq(True)].drop(columns="is_calculated")

            factor.to_parquet(f"{self.output_dir}/{factor_name}.parquet", index=False)
            del factor, factor_by_code, rolling
