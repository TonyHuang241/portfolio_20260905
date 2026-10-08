import os

import pandas as pd

from trading_strategies.strategies_evaluation.strategy_evaluation import StrategyEvaluation


class MarketTiming:
    def __init__(self, config):
        self.factor_data_dir = config["factor_data_dir"]
        self.stock_daily_dir = config["stock_daily_dir"]
        self.backtest_data_dir = config["backtest_data_dir"]
        self.limit_price_dir = config["limit_price_dir"]
        self.stock_info_path = config["stock_info_path"]
        self.output_dir = os.path.join(config["output_dir"], type(self).__name__)
        self.start_date = config["start_date"]
        self.stock_board = config["stock_board"]
        self.pool_factor = config["pool_factor"]
        self.pool_ascending = config["pool_ascending"]
        self.pool_size = config["pool_size"]
        self.select_factor = config["select_factor"]
        self.select_ascending = config["select_ascending"]
        self.select_size = config["select_size"]
        self.timing_window = config["timing_window"]
        self.fee_rate = config["fee_rate"]

    def select_stocks(self):
        """按板块筛选后，每日取 pool_factor 排名前 pool_size 只作为股票池，并在池内按 select_factor 排名。"""
        stocks = pd.read_parquet(os.path.join(self.factor_data_dir, f"{self.pool_factor.split('_')[0]}.parquet"), columns=["code", "date", self.pool_factor])
        stocks = stocks.merge(pd.read_parquet(os.path.join(self.factor_data_dir, f"{self.select_factor.split('_')[0]}.parquet"), columns=["code", "date", self.select_factor]), on=["code", "date"], how="left")
        # main board 只保留沪深主板，all 不筛选。
        if self.stock_board == "main board":
            stocks = stocks.loc[stocks["code"].str.startswith(("000", "001", "002", "003", "600", "601", "603", "605"))]
        elif self.stock_board != "all":
            raise ValueError(f"stock_board must be 'main board' or 'all', got {self.stock_board!r}")

        stocks["pool_rank"] = stocks.groupby("date")[self.pool_factor].rank(ascending=self.pool_ascending, method="first")
        stocks = stocks.loc[stocks["pool_rank"].le(self.pool_size)]
        stocks["select_rank"] = stocks.groupby("date")[self.select_factor].rank(ascending=self.select_ascending, method="first")
        return stocks

    def time_market(self, stocks):
        """用前一交易日的股票池在当日的等权收盘收益构建池子净值，当日净值高于 timing_window 日均线时下一交易日持有，否则空仓。"""
        daily = pd.read_parquet(os.path.join(self.stock_daily_dir, "daily_stock_data.parquet"), columns=["code", "date", "close"])
        daily = daily.sort_values(["code", "date"])
        daily["return"] = daily.groupby("code")["close"].pct_change(fill_method=None)

        # 信号日收盘后确定持仓，下一交易日 10 点按 trade_date 交易。
        trade_dates = sorted(daily["date"].unique())
        stocks["trade_date"] = stocks["date"].map(dict(zip(trade_dates[:-1], trade_dates[1:])))
        timing = stocks.merge(daily.rename(columns={"date": "trade_date"}), on=["code", "trade_date"])
        timing = timing.groupby("trade_date")["return"].mean().to_frame()
        timing["nav"] = (1 + timing["return"]).cumprod()
        timing["hold"] = timing["nav"].gt(timing["nav"].rolling(self.timing_window).mean())
        stocks["hold"] = stocks["date"].map(timing["hold"]).eq(True)
        return stocks

    def run(self):
        """选股、择时后保存每日候选股票清单（含下一交易日，hold 为 False 表示空仓），并自动生成策略评估报告。"""
        stocks = self.select_stocks()
        stocks = self.time_market(stocks)
        stocks = stocks.loc[stocks["select_rank"].le(self.select_size) & stocks["date"].ge(self.start_date), ["date", "trade_date", "code", self.pool_factor, self.select_factor, "hold"]]
        stocks = stocks.sort_values(["date", self.select_factor], ascending=[True, self.select_ascending])
        # 补充股票中文简称，去掉简称中用于对齐的空格（如“万  科Ａ”）。
        stock_info = pd.read_csv(self.stock_info_path, usecols=["code", "name"], dtype=str)
        stock_info["code"] = stock_info["code"].str.split(".").str[0]
        stock_info["name"] = stock_info["name"].str.replace(" ", "")
        stocks.insert(3, "name", stocks["code"].map(stock_info.set_index("code")["name"]))
        os.makedirs(self.output_dir, exist_ok=True)
        stocks.to_csv(os.path.join(self.output_dir, f"{type(self).__name__}Holdings.csv"), index=False, encoding="utf-8-sig")
        StrategyEvaluation(stocks, self.backtest_data_dir, self.limit_price_dir, self.output_dir, self.fee_rate).evaluate()
        return stocks
