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

    @staticmethod
    def _format_codes(codes: pd.Series) -> pd.Series:
        # 先处理不重复的代码，再映射回各行；对全量数据 str.split 会逐行生成列表，非常慢。
        unique_codes = codes.dropna().drop_duplicates()
        formatted_codes = unique_codes.str.split(".", n=1).str[0].str.zfill(6)
        return codes.map(dict(zip(unique_codes, formatted_codes)))

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
        limit_price["code"] = self._format_codes(limit_price["code"])
        # 日级日期统一使用 date 列及 YYYYMMDD 字符串，与因子和评估数据一致。
        limit_price["date"] = self._format_dates(limit_price["date"])
        limit_price_dir = Path(self.output_dir) / "limit_price"
        limit_price_dir.mkdir(parents=True, exist_ok=True)
        limit_price.to_parquet(limit_price_dir / "limit_price.parquet", index=False)

        stock_st_status = pd.read_csv(Path(self.new_data_dir) / "stock_st_status.csv", dtype={"code": str, "date": str})
        stock_st_status["code"] = self._format_codes(stock_st_status["code"])
        stock_st_status["date"] = self._format_dates(stock_st_status["date"])
        stock_st_status_dir = Path(self.output_dir) / "stock_st_status"
        stock_st_status_dir.mkdir(parents=True, exist_ok=True)
        stock_st_status.to_parquet(stock_st_status_dir / "stock_st_status.parquet", index=False)

        mkcap = pd.read_csv(Path(self.new_data_dir) / "mkcap.csv", dtype={"code": str, "trading_day": str})
        mkcap["code"] = self._format_codes(mkcap["code"])
        mkcap = mkcap.rename(columns={"trading_day": "date", "total_value": "mkcap"})
        mkcap["date"] = self._format_dates(mkcap["date"])
        fundamentals_dir = Path(self.output_dir) / "fundamentals"
        fundamentals_dir.mkdir(parents=True, exist_ok=True)
        mkcap.to_parquet(fundamentals_dir / "mkcap.parquet", index=False)

        # 每只股票上月最后一个交易日的市值，记在当月第一天。
        mkcap_monthly = mkcap.sort_values("date")
        mkcap_monthly["date"] = self._format_dates(pd.to_datetime(mkcap_monthly["date"], format="%Y%m%d") + pd.offsets.MonthBegin(1))
        mkcap_monthly = mkcap_monthly.drop_duplicates(subset=["code", "date"], keep="last")
        mkcap_monthly.to_parquet(fundamentals_dir / "mkcap_monthly.parquet", index=False)

        # 季度财务数据：公告日记为 date，与日线对齐；报告期见 year、quarter 列。
        financial_data = pd.read_csv(Path(self.new_data_dir) / "fundamentals_quarterly.csv", dtype={"secu_code": str, "publ_date": str})
        financial_data = financial_data.rename(columns={"secu_code": "code", "publ_date": "date"})
        financial_data["code"] = self._format_codes(financial_data["code"])
        financial_data["date"] = self._format_dates(financial_data["date"])

        # 流量变量为年内累计值，换成 TTM：Q1-Q3 为 上年年报 + 本期累计 - 上年同期累计，Q4 为当年年报；时点变量保留期末原值。
        flow_columns = ["total_operating_revenue", "operating_revenue", "operating_cost", "operating_profit", "total_profit", "net_profit", "np_parent_company_owners", "r_and_d", "net_operate_cash_flow", "net_invest_cash_flow", "net_finance_cash_flow", "cash_equivalent_increase"]
        financial_data["previous_year"] = financial_data["year"] - 1
        financial_data["annual_quarter"] = 4
        financial_data = financial_data.join(financial_data.set_index(["code", "year", "quarter"])[flow_columns].add_suffix("_last_year"), on=["code", "previous_year", "quarter"])
        financial_data = financial_data.join(financial_data.set_index(["code", "year", "quarter"])[flow_columns + ["date"]].add_suffix("_last_annual"), on=["code", "previous_year", "annual_quarter"])
        for column in flow_columns:
            financial_data[f"{column}_ttm"] = financial_data[f"{column}_last_annual"] + financial_data[column] - financial_data[f"{column}_last_year"]
            financial_data[f"{column}_ttm"] = financial_data[f"{column}_ttm"].where(financial_data["quarter"].ne(4), financial_data[column])
            financial_data = financial_data.drop(columns=[column, f"{column}_last_year", f"{column}_last_annual"])

        # 上年年报晚于本期公告时（如 2020 年年报延期），TTM 要到年报公告后才可得，date 顺延至年报公告日。
        financial_data["date"] = financial_data["date"].where(financial_data["quarter"].eq(4) | ~financial_data["date"].lt(financial_data["date_last_annual"]), financial_data["date_last_annual"])
        financial_data = financial_data.drop(columns=["end_date", "previous_year", "annual_quarter", "date_last_annual"])

        # 没有公告日的报告期无法确定可用时间，剔除；同一天公告多期报告（多为年报与一季报同日）时只保留最新报告期，使 code、date 唯一。
        financial_data = financial_data.loc[financial_data["date"].notna()]
        financial_data = financial_data.sort_values(["code", "date", "year", "quarter"])
        financial_data = financial_data.drop_duplicates(["code", "date"], keep="last")
        financial_data.to_parquet(fundamentals_dir / "financial_data.parquet", index=False)

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
        # 全天成交量为 0（成交量缺失按 0 计）判定为停牌；当天没有该股票数据的，在因子构建补齐交易日后同样视为停牌。
        daily_stock["is_suspended"] = daily_stock["volume"].eq(0)
        result = daily_minutes.loc[daily_minutes["trade_time"].eq(trade_date + pd.Timedelta(hours=10)) & daily_minutes["open"].ne(0)].copy()
        result["is_constant_close"] = result["code"].map(daily_close["min"].eq(daily_close["max"]))
        result["is_suspended"] = result["code"].map(daily_stock.set_index("code")["is_suspended"])
        del daily_minutes, daily_close
        gc.collect()
        return result, daily_stock

    def _calculate_adj_factors(self, daily_stock_data):
        """由除权字段和除权日前一交易日的原始收盘价计算累计等比后复权因子，每次除权一行。"""
        exrights = pd.read_csv(Path(self.output_dir) / "adj_factors" / "stock_exrights.csv", usecols=["date", "code", "allotted_ps", "rationed_ps", "rationed_px", "bonus_ps"], dtype={"date": str, "code": str})
        exrights["code"] = self._format_codes(exrights["code"])
        exrights["__adj_date"] = pd.to_datetime(exrights["date"], format="%Y%m%d")
        exrights = exrights.sort_values("__adj_date")

        # 取除权日之前最后一个有收盘价的交易日，停牌跨过除权日时即停牌前的收盘价。
        daily_close = daily_stock_data.loc[daily_stock_data["close"].gt(0), ["code", "date", "close"]].rename(columns={"close": "previous_close"})
        daily_close["__adj_date"] = pd.to_datetime(daily_close["date"], format="%Y%m%d")
        daily_close = daily_close.drop(columns="date").sort_values("__adj_date")
        exrights = pd.merge_asof(exrights, daily_close, on="__adj_date", by="code", direction="backward", allow_exact_matches=False)

        # 单次因子 = 前收盘 / 除权参考价，参考价 = (前收盘 - 每股派息 + 配股价 × 配股比例) / (1 + 送转比例 + 配股比例)。
        # 早于数据起点的除权没有前收盘，对所有已有价格是同一个常数，不影响收益，取 1。
        exrights["adj_factor"] = exrights["previous_close"] * (1 + exrights["allotted_ps"] + exrights["rationed_ps"]) / (exrights["previous_close"] - exrights["bonus_ps"] + exrights["rationed_px"] * exrights["rationed_ps"])
        exrights["adj_factor"] = exrights["adj_factor"].fillna(1)
        exrights = exrights.sort_values(["code", "__adj_date"])
        exrights["adj_factor"] = exrights.groupby("code")["adj_factor"].cumprod()
        return exrights[["code", "__adj_date", "adj_factor"]].sort_values("__adj_date")

    @staticmethod
    def _adjust_prices(data, adj_factors, date_column):
        """按当日已生效的累计等比因子后复权：原始价格 × adj_factor，相邻复权价之比即真实收益。"""
        data = data.copy()
        data["__adj_date"] = pd.to_datetime(data[date_column]).dt.normalize()
        data = data.sort_values("__adj_date")
        data = pd.merge_asof(data, adj_factors, on="__adj_date", by="code", direction="backward")
        data["adj_factor"] = data["adj_factor"].fillna(1)
        for column in data.columns.intersection(["open", "high", "low", "close", "pre_close", "price"]):
            data[column] = data[column] * data["adj_factor"]
        return data.drop(columns="__adj_date")

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
            # 已有数据除以原因子还原为原始价格，与新数据一起按最新除权信息重新复权。
            for column in existing_data.columns.intersection(["open", "high", "low", "close", "pre_close", "price"]):
                existing_data[column] = existing_data[column] / existing_data["adj_factor"]
            tables.append(existing_data.drop(columns="adj_factor"))

        if self.update_all != 1 and daily_stock_file.exists():
            existing_daily = pd.read_parquet(daily_stock_file)
            existing_daily = existing_daily.loc[existing_daily["date"] >= start_date.strftime("%Y%m%d")]
            existing_dates &= set(pd.to_datetime(existing_daily["date"].drop_duplicates(), format="%Y%m%d"))
            for column in existing_daily.columns.intersection(["open", "high", "low", "close"]):
                existing_daily[column] = existing_daily[column] / existing_daily["adj_factor"]
            daily_tables.append(existing_daily.drop(columns=["adj_factor", "mkcap"], errors="ignore"))
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
                results = list(tqdm(results, total=len(daily_sources), desc="Preparing backtest data", unit="day"))
            tables.extend([backtest for backtest, _ in results])
            daily_tables.extend([daily_stock for _, daily_stock in results])

        # 合并新旧日线（均为原始价格），按日期和股票代码去重、排序。
        daily_stock_data = pd.concat(daily_tables, ignore_index=True)
        daily_stock_data["code"] = self._format_codes(daily_stock_data["code"].astype(str))
        daily_stock_data = daily_stock_data.drop_duplicates(subset=["date", "code"], keep="last")

        # 按最新的 mkcap.parquet 为新旧日线统一匹配当日总市值；市值源文件可能有重复行，先去重，缺失市值保留为空。
        mkcap = pd.read_parquet(Path(self.output_dir) / "fundamentals" / "mkcap.parquet", columns=["code", "date", "mkcap"])
        mkcap = mkcap.drop_duplicates(subset=["code", "date"], keep="last")
        daily_stock_data = daily_stock_data.merge(mkcap, on=["code", "date"], how="left")

        # 由原始收盘价计算等比因子，日线和 10 点数据用同一套因子统一复权一次。
        adj_factors = self._calculate_adj_factors(daily_stock_data)
        daily_stock_data = self._adjust_prices(daily_stock_data, adj_factors, "date")
        daily_stock_data = daily_stock_data.sort_values(["date", "code"])
        # 交易日序号：全部日线日期按先后连续编号，相邻交易日相差 1，个股序号不连续即中间有停牌；10 点数据使用同一套编号。
        daily_stock_data["trade_day"] = daily_stock_data["date"].rank(method="dense").astype(int)
        daily_stock_data.to_parquet(daily_stock_file, index=False)

        # 合并新旧数据，按时间和股票代码去重、复权、排序后保存。
        backtest_data = pd.concat(tables, ignore_index=True)
        backtest_data["code"] = self._format_codes(backtest_data["code"].astype(str))
        backtest_data = backtest_data.drop_duplicates(subset=["trade_time", "code"], keep="last")
        backtest_data = self._adjust_prices(backtest_data, adj_factors, "trade_time")
        backtest_data = backtest_data.sort_values(["trade_time", "code"])
        # 原始数据的 date 列部分为空，统一由 trade_time 生成原始日期，再按日期匹配日线的交易日序号。
        trade_days = daily_stock_data.drop_duplicates("date").set_index("date")["trade_day"]
        backtest_data["date"] = self._format_dates(backtest_data["trade_time"].dt.normalize())
        backtest_data["trade_day"] = backtest_data["date"].map(trade_days)

        # 停牌日保留价格供回测估值，但不计入连续交易天数（记为空）；在非停牌样本上按股票划分连续区间，相邻记录的交易日序号相差超过 10（中间停牌 10 个交易日及以上）时重新计数。
        trading_data = backtest_data.loc[~backtest_data["is_suspended"].eq(True), ["code", "trade_day"]]
        trading_data["consecutive_trading_days"] = trading_data.groupby("code")["trade_day"].diff().gt(10)
        trading_data["consecutive_trading_days"] = trading_data.groupby("code")["consecutive_trading_days"].cumsum()
        backtest_data["consecutive_trading_days"] = trading_data.groupby(["code", "consecutive_trading_days"]).cumcount() + 1
        backtest_data.to_parquet(backtest_file, index=False)
