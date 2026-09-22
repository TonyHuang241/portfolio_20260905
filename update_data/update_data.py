import gc
from io import BytesIO
from multiprocessing import Pool
from pathlib import Path
from zipfile import ZipFile

import pandas as pd
from tqdm import tqdm


class DataUpdater:
    def __init__(self, config: dict):
        self.new_data_dir = config["new_data_dir"]
        self.output_dir = config["output_dir"]
        self.start_date = config["start_date"]
        self.processes = config["processes"]
        self.update_all = config["update_all"]

    @staticmethod
    def _format_dates(dates: pd.Series) -> pd.Series:
        # 先转换不重复的日期，再映射回各行，避免对全量数据重复格式化。
        unique_dates = dates.dropna().drop_duplicates()
        formatted_dates = pd.to_datetime(unique_dates).dt.strftime("%Y%m%d")
        return dates.map(dict(zip(unique_dates, formatted_dates)))

    def integrate_data(self):
        minutes = pd.read_csv(Path(self.new_data_dir) / "daily_minutes.csv", dtype={"code": str})
        exrights = pd.read_csv(Path(self.new_data_dir) / "stock_exrights.csv", dtype={"code": str})
        exrights["code"] = exrights["code"].str.split(".", n=1).str[0].str.zfill(6)
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
        limit_price["code"] = limit_price["code"].str.split(".", n=1).str[0].str.zfill(6)
        # 日级日期统一使用 date 列及 YYYYMMDD 字符串，与因子和评估数据一致。
        limit_price["date"] = self._format_dates(limit_price["date"])
        limit_price_dir = Path(self.output_dir) / "limit_price"
        limit_price_dir.mkdir(parents=True, exist_ok=True)
        limit_price.to_parquet(limit_price_dir / "limit_price.parquet", index=False)

        stock_st_status = pd.read_csv(Path(self.new_data_dir) / "stock_st_status.csv", dtype={"code": str, "date": str})
        stock_st_status["code"] = stock_st_status["code"].str.split(".", n=1).str[0].str.zfill(6)
        stock_st_status["date"] = self._format_dates(stock_st_status["date"])
        stock_st_status_dir = Path(self.output_dir) / "stock_st_status"
        stock_st_status_dir.mkdir(parents=True, exist_ok=True)
        stock_st_status.to_parquet(stock_st_status_dir / "stock_st_status.parquet", index=False)

        mkcap = pd.read_csv(Path(self.new_data_dir) / "mkcap.csv", dtype={"code": str, "trading_day": str})
        mkcap["code"] = mkcap["code"].str.split(".", n=1).str[0].str.zfill(6)
        mkcap = mkcap.rename(columns={"trading_day": "date", "total_value": "mkcap"})
        mkcap["date"] = self._format_dates(mkcap["date"])
        fundamentals_dir = Path(self.output_dir) / "fundamentals"
        fundamentals_dir.mkdir(parents=True, exist_ok=True)
        mkcap.to_parquet(fundamentals_dir / "mkcap.parquet", index=False)

        # 每只股票上月最后一个交易日的市值，记在当月第一天。
        mkcap_monthly = mkcap.sort_values("date")
        mkcap_monthly["date"] = (pd.to_datetime(mkcap_monthly["date"], format="%Y%m%d") + pd.offsets.MonthBegin(1)).dt.strftime("%Y%m%d")
        mkcap_monthly = mkcap_monthly.drop_duplicates(subset=["code", "date"], keep="last")
        mkcap_monthly.to_parquet(fundamentals_dir / "mkcap_monthly.parquet", index=False)

    @staticmethod
    def _prepare_daily_backtest_data(item):
        trade_date, (source, member) = item
        if member is None:
            daily_minutes = pd.read_parquet(source)
        else:
            with ZipFile(source) as archive:
                # Parquet 会随机读取；一次性解压，避免 ZIP 流回退时重复解压。
                with BytesIO(archive.read(member)) as buffer:
                    daily_minutes = pd.read_parquet(buffer)

        daily_minutes["trade_time"] = pd.to_datetime(daily_minutes["trade_time"])
        daily_minutes = daily_minutes.sort_values("trade_time", kind="stable")
        daily_stock = daily_minutes.groupby("code", sort=False, as_index=False).agg(
            open=("open", "first"), high=("high", "max"), low=("low", "min"), close=("close", "last"),
            volume=("volume" if "volume" in daily_minutes.columns else "vol", "sum"),
            money=("money" if "money" in daily_minutes.columns else "amount", "sum"),
        )
        daily_stock.insert(0, "date", trade_date.strftime("%Y%m%d"))
        # 最低、最高收盘价相等即全天价格恒定，避免计算标准差及其浮点误差。
        daily_close = daily_minutes.groupby("code", sort=False)["close"].agg(["min", "max"])
        result = daily_minutes.loc[daily_minutes["trade_time"].eq(trade_date + pd.Timedelta(hours=10)) & daily_minutes["open"].ne(0)].copy()
        result["is_constant_close"] = result["code"].map(daily_close["min"].eq(daily_close["max"]))
        del daily_minutes, daily_close
        gc.collect()
        return result, daily_stock

    def generate_backtest_data(self):
        backtest_dir = Path(self.output_dir) / "stock_daily"
        backtest_dir.mkdir(parents=True, exist_ok=True)
        backtest_file = backtest_dir / "daily_stock_data_10am.parquet"
        daily_stock_file = backtest_dir / "daily_stock_data.parquet"
        start_date = pd.Timestamp(self.start_date).normalize()
        tables = []
        daily_tables = []
        existing_dates = set()

        # 增量更新时读取已有结果；全量更新从起始日期重新计算。
        if self.update_all != 1 and backtest_file.exists():
            existing_data = pd.read_parquet(backtest_file)
            existing_data["trade_time"] = pd.to_datetime(existing_data["trade_time"])
            existing_data = existing_data.loc[(existing_data["trade_time"] >= start_date) & existing_data["open"].ne(0)]
            existing_dates = set(existing_data["trade_time"].dt.normalize().drop_duplicates())
            tables.append(existing_data)

        if self.update_all != 1 and daily_stock_file.exists():
            existing_daily = pd.read_parquet(daily_stock_file)
            existing_daily = existing_daily.loc[existing_daily["date"] >= start_date.strftime("%Y%m%d")]
            existing_dates &= set(pd.to_datetime(existing_daily["date"].drop_duplicates(), format="%Y%m%d"))
            daily_tables.append(existing_daily)
        else:
            existing_dates.clear()

        # 两个结果均已有的日期才跳过；从 ZIP 和普通文件夹中收集缺失日期，普通文件优先使用。
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

        # 增量更新没有待补日期时，避免重写。
        if tables and daily_tables and not daily_sources:
            return

        # 同时生成日频 OHLC 和恰好 10:00 且开盘价不为 0 的回测记录。
        if daily_sources:
            with Pool(processes=self.processes) as pool:
                results = pool.imap(self._prepare_daily_backtest_data, sorted(daily_sources.items()), chunksize=1)
                for backtest, daily_stock in tqdm(results, total=len(daily_sources), desc="Preparing backtest data", unit="day"):
                    tables.append(backtest)
                    daily_tables.append(daily_stock)

        # 合并新旧数据，按时间和股票代码去重、排序后保存。
        backtest_data = pd.concat(tables, ignore_index=True)
        backtest_data["code"] = backtest_data["code"].astype(str).str.split(".", n=1).str[0].str.zfill(6)
        backtest_data = backtest_data.drop_duplicates(subset=["trade_time", "code"], keep="last")
        backtest_data = backtest_data.sort_values(["trade_time", "code"])

        # 在过滤后的样本上按股票划分连续区间，日期间隔超过 15 个自然日时重新计数。
        backtest_data["consecutive_trading_days"] = backtest_data.groupby("code")["trade_time"].diff().dt.days.gt(15)
        backtest_data["consecutive_trading_days"] = backtest_data.groupby("code")["consecutive_trading_days"].cumsum()
        backtest_data["consecutive_trading_days"] = backtest_data.groupby(["code", "consecutive_trading_days"]).cumcount() + 1
        backtest_data.to_parquet(backtest_file, index=False)

        daily_stock_data = pd.concat(daily_tables, ignore_index=True)
        daily_stock_data["code"] = daily_stock_data["code"].astype(str).str.split(".", n=1).str[0].str.zfill(6)
        daily_stock_data = daily_stock_data.drop_duplicates(subset=["date", "code"], keep="last")
        daily_stock_data = daily_stock_data.sort_values(["date", "code"])
        daily_stock_data.to_parquet(daily_stock_file, index=False)
