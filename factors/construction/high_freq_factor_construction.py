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
        self.limit_price_dir = config["limit_price_dir"]
        self.stock_st_status_dir = config["stock_st_status_dir"]
        self.adj_factor_dir = config["adj_factor_dir"]
        self.output_dir = config["output_dir"]
        self.factor_calc_dir = config["factor_calc_dir"]

        self.start_date = config["start_date"]
        self.end_date = config["end_date"]
        self.update_all = config["update_all"]

        self._update_trade_date()
        self._regist_factors()

        pass

    def _update_trade_date(self):
        """获取分钟数据中存在的全部交易日期"""
        trade_dates = set()
        for year_entry in os.listdir(self.stock_minutes_dir):
            year_path = os.path.join(self.stock_minutes_dir, year_entry)
            if os.path.isdir(year_path):
                file_names = os.listdir(year_path)
            elif year_entry.endswith(".zip"):
                with ZipFile(year_path) as archive:
                    file_names = archive.namelist()
            else:
                continue

            for file_name in file_names:
                trade_date, extension = os.path.splitext(os.path.basename(file_name))
                if extension == ".parquet" and len(trade_date) == 8 and trade_date.isdigit():
                    trade_dates.add(trade_date)

        self.trade_date = sorted(trade_dates)

    def _regist_factors(self):
        """注册项目 factor_calculation 包中的因子，文件名、类名与因子名须一致。"""
        self.factors = [
            file[:-3] for file in sorted(os.listdir(self.factor_calc_dir))
            if not file.startswith("_") and file.endswith(".py") and os.path.isfile(os.path.join(self.factor_calc_dir, file))
        ]

    @staticmethod
    def _read_minutes_file(stock_minutes_dir, trade_date):
        """读取日度股票分钟回报数据"""
        year_path = os.path.join(stock_minutes_dir, trade_date[:4])
        file_name = f"{trade_date}.parquet"
        minutes_path = os.path.join(year_path, file_name)
        if os.path.isfile(minutes_path):
            return pd.read_parquet(minutes_path)

        with ZipFile(f"{year_path}.zip") as archive:
            # Parquet 会随机读取文件，先一次性解压，避免 ZIP 文件流反复解压。
            return pd.read_parquet(BytesIO(archive.read(file_name)))

    @staticmethod
    def _calculate_factor(factor_name, minutes, daily_limit_price):
        """计算单个因子的原始值，供进程池调用。"""
        module = import_module(f"factors.construction.factor_calculation.{factor_name}")
        return getattr(module, factor_name)(minutes=minutes, limit_price=daily_limit_price).calculate()

    def _calculate_trade_date(self, task):
        """读取并清洗一天的行情，顺序计算当天缺失的因子，供进程池调用。"""
        trade_date, pending_factors, daily_st_status, daily_limit_price = task
        minutes = self._read_minutes_file(self.stock_minutes_dir, trade_date)
        minutes = minutes.rename(columns={"amount": "money", "vol": "volume"})
        minutes["trade_time"] = pd.to_datetime(minutes["trade_time"])
        minutes["date"] = minutes["trade_time"].dt.normalize()
        # 只格式化不同日期，避免对每日上百万条分钟记录重复转换字符串。
        minutes["date"] = minutes["date"].map({date: date.strftime("%Y%m%d") for date in minutes["date"].dropna().unique()})

        # 剔除当天处于 ST 状态的股票。
        minutes = minutes.loc[~minutes["code"].isin(daily_st_status.loc[daily_st_status["is_st"], "code"])]

        # 任一条开盘价为零或全天收盘价恒定，则清空该股票当天全部行情，保留代码和时间。
        invalid_day = minutes["open"].eq(0).groupby([minutes["code"], minutes["date"]]).transform("any")
        # 用唯一值数量判断价格恒定，避免浮点误差使标准差不严格为零。
        invalid_day |= minutes.groupby(["code", "date"])["close"].transform("nunique").eq(1)
        minutes.loc[invalid_day, minutes.columns.difference(["code", "trade_time", "date"])] = np.nan

        return trade_date, {name: self._calculate_factor(name, minutes, daily_limit_price) for name in pending_factors}

    def update_factors(self):
        """按交易日并行更新因子数据。"""
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

        limit_price = pd.read_parquet(os.path.join(self.limit_price_dir, "limit_price.parquet"))
        limit_price_by_date = limit_price.groupby("date")

        stock_st_status = pd.read_parquet(os.path.join(self.stock_st_status_dir, "stock_st_status.parquet"))
        stock_st_status_by_date = stock_st_status.groupby("date")

        # 按交易日并行，分钟数据在子进程内读取，避免在进程间传输整日行情。
        pending_dates = [date for date in trade_dates if date not in common_exist_dates]
        if pending_dates:
            tasks = (
                (date, [name for name in self.factors if date not in exist_dates[name]],
                 stock_st_status_by_date.get_group(date), limit_price_by_date.get_group(date))
                for date in pending_dates
            )
            with Pool(processes=min(2, len(pending_dates))) as pool:
                with tqdm(total=len(pending_dates), desc="Updating raw factor values", unit="trading day") as trade_date_progress:
                    for trade_date, daily_factors in pool.imap_unordered(self._calculate_trade_date, tasks):
                        for factor_name, daily_factor in daily_factors.items():
                            factor_data[factor_name].append(daily_factor)
                        trade_date_progress.set_postfix_str(f"Completed date: {trade_date}")
                        trade_date_progress.update()

        # 逐个计算并保存，避免同时保留全部因子的滚动指标。
        os.makedirs(self.output_dir, exist_ok=True)
        for factor_name in tqdm(self.factors, desc="Updating derived factor values", unit="factor"):
            factor = pd.concat(factor_data.pop(factor_name), ignore_index=True)
            factor = factor.drop_duplicates(["code", "date"], keep="last")
            factor = factor.sort_values(["code", "date"]).reset_index(drop=True)
            factor_by_code = factor.groupby("code")[factor_name]
            for window in (20, 60, 120):
                rolling = factor_by_code.rolling(window)
                factor[f"{factor_name}_{window}_m"] = rolling.mean().droplevel(0).reindex(factor.index)
                factor[f"{factor_name}_{window}_std"] = rolling.std().droplevel(0).reindex(factor.index)

            factor.to_parquet(f"{self.output_dir}/{factor_name}.parquet", index=False)
            del factor, factor_by_code, rolling
