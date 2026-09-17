from contextlib import ExitStack
from pathlib import Path
from zipfile import ZipFile

import pandas as pd
from tqdm import tqdm


class DataUpdater:
    def __init__(self, config: dict):
        self.new_data_dir = config["new_data_dir"]
        self.output_dir = config["output_dir"]
        self.start_date = config["start_date"]

    def integrate_data(self):
        minutes = pd.read_csv(Path(self.new_data_dir) / "daily_minutes.csv", dtype={"code": str})
        exrights = pd.read_csv(Path(self.new_data_dir) / "stock_exrights.csv", dtype={"code": str})
        minutes = minutes.rename(columns={"Unnamed: 0": "trade_time", "datetime": "trade_time"})

        minutes["trade_time"] = pd.to_datetime(minutes["trade_time"])

        daily_groups = minutes.groupby(minutes["trade_time"].dt.normalize(), sort=True)
        progress = tqdm(daily_groups, total=daily_groups.ngroups, desc="Updating daily data", unit="day")
        for trade_date, daily_minutes in progress:
            progress.set_description(f"Updating data for {trade_date:%Y-%m-%d}")

            year_dir = Path(self.output_dir) / "stock_minutes" / f"{trade_date.year:04d}"
            year_dir.mkdir(parents=True, exist_ok=True)
            daily_minutes.to_parquet(year_dir / f"{trade_date:%Y%m%d}.parquet", index=False)

        adj_factors_dir = Path(self.output_dir) / "adj_factors"
        adj_factors_dir.mkdir(parents=True, exist_ok=True)
        exrights.to_csv(adj_factors_dir / "stock_exrights.csv", index=False, encoding="utf-8-sig")

        limit_price = pd.read_csv(Path(self.new_data_dir) / "limit_price.csv", dtype={"code": str, "date": str})
        # 日级日期统一使用 date 列及 YYYYMMDD 字符串，与因子和评估数据一致。
        limit_price["date"] = pd.to_datetime(limit_price["date"]).dt.strftime("%Y%m%d")
        limit_price_dir = Path(self.output_dir) / "limit_price"
        limit_price_dir.mkdir(parents=True, exist_ok=True)
        limit_price.to_parquet(limit_price_dir / "limit_price.parquet", index=False)

        stock_st_status = pd.read_csv(Path(self.new_data_dir) / "stock_st_status.csv", dtype={"code": str, "date": str})
        stock_st_status["date"] = pd.to_datetime(stock_st_status["date"]).dt.strftime("%Y%m%d")
        stock_st_status_dir = Path(self.output_dir) / "stock_st_status"
        stock_st_status_dir.mkdir(parents=True, exist_ok=True)
        stock_st_status.to_parquet(stock_st_status_dir / "stock_st_status.parquet", index=False)

    def generate_backtest_data(self):
        backtest_dir = Path(self.output_dir) / "backtest_data"
        backtest_dir.mkdir(parents=True, exist_ok=True)
        backtest_file = backtest_dir / "backtest10am.parquet"
        start_date = pd.Timestamp(self.start_date).normalize()
        tables = []
        existing_dates = set()

        # 读取已有结果，记录起始日期之后已覆盖的交易日（含起始日）。
        if backtest_file.exists():
            existing_data = pd.read_parquet(backtest_file)
            existing_data["trade_time"] = pd.to_datetime(existing_data["trade_time"])
            existing_data = existing_data.loc[existing_data["trade_time"] >= start_date]
            existing_dates = set(existing_data["trade_time"].dt.normalize())
            tables.append(existing_data)

        # 从 ZIP 和普通文件夹中收集缺失日期，普通文件优先使用。
        minutes_dir = Path(self.output_dir) / "stock_minutes"
        daily_sources = {}
        for archive in sorted(minutes_dir.glob("*.zip")):
            with ZipFile(archive) as zip_file:
                for member in zip_file.namelist():
                    if Path(member).suffix != ".parquet":
                        continue
                    trade_date = pd.to_datetime(Path(member).stem, format="%Y%m%d")
                    if trade_date >= start_date and trade_date not in existing_dates:
                        daily_sources[trade_date] = (archive, member)

        for daily_file in sorted(minutes_dir.glob("*/*.parquet")):
            trade_date = pd.to_datetime(daily_file.stem, format="%Y%m%d")
            if trade_date >= start_date and trade_date not in existing_dates:
                daily_sources[trade_date] = (daily_file, None)

        # 逐日提取恰好 10:00 的记录，保留全部数据列。
        with ExitStack() as stack:
            archive_paths = sorted({source for source, member in daily_sources.values() if member is not None})
            archives = {archive: stack.enter_context(ZipFile(archive)) for archive in archive_paths}
            for _, (source, member) in tqdm(sorted(daily_sources.items()), desc="Preparing 10am backtest data", unit="day"):
                if member is None:
                    daily_minutes = pd.read_parquet(source)
                else:
                    with archives[source].open(member) as parquet_file:
                        daily_minutes = pd.read_parquet(parquet_file)

                daily_minutes["trade_time"] = pd.to_datetime(daily_minutes["trade_time"])
                time_of_day = daily_minutes["trade_time"] - daily_minutes["trade_time"].dt.normalize()
                tables.append(daily_minutes.loc[time_of_day == pd.Timedelta(hours=10)])

        # 合并新旧数据，按时间和股票代码去重、排序后保存。
        backtest_data = pd.concat(tables, ignore_index=True)
        backtest_data = backtest_data.drop_duplicates(subset=["trade_time", "code"], keep="last")
        backtest_data = backtest_data.sort_values(["trade_time", "code"])
        backtest_data.to_parquet(backtest_file, index=False)
