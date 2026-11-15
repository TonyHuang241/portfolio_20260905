import os
from importlib import import_module

import pandas as pd

from trading_strategies.strategies_evaluation.strategy_backtest import StrategyBacktest
from trading_strategies.strategies_evaluation.strategy_report import StrategyReport


class StrategyEvaluation:
    def __init__(self, config):
        self.config = config
        self.strategy = config["strategy"]
        self.holdings_path = config["holdings_path"]
        self.output_dir = config["output_dir"]
        self.backtest_data_dir = config["backtest_data_dir"]
        self.limit_price_dir = config["limit_price_dir"]
        self.fee_rate = config["fee_rate"]

    def run(self):
        """holdings_path 为空时运行 strategy 指定的策略生成持仓清单，否则直接读取该清单；然后回测并生成报告。"""
        if self.holdings_path:
            holdings = pd.read_csv(self.holdings_path, dtype={"code": str, "date": str, "trade_date": str})
            # 清单没有 strategy 列时，用文件名去掉 Holdings 后缀作为策略名。
            if "strategy" not in holdings.columns:
                holdings.insert(0, "strategy", os.path.basename(self.holdings_path).removesuffix(".csv").removesuffix("Holdings"))
            # 报告从输出目录读取持仓清单，清单不在输出目录时复制一份过去。
            output_path = os.path.join(self.output_dir, holdings["strategy"].iloc[0], f"{holdings['strategy'].iloc[0]}Holdings.csv")
            if os.path.abspath(self.holdings_path) != os.path.abspath(output_path):
                os.makedirs(os.path.dirname(output_path), exist_ok=True)
                holdings.to_csv(output_path, index=False, encoding="utf-8-sig")
        else:
            # 策略按文件名加载，文件名即类名，新增策略不需要改动这里。
            module = import_module(f"trading_strategies.strategies.{self.strategy}")
            holdings = getattr(module, self.strategy)(self.config).run()
        StrategyBacktest(holdings, self.backtest_data_dir, self.limit_price_dir, self.output_dir, self.fee_rate).run()
        StrategyReport(self.output_dir, holdings["strategy"].iloc[0], self.fee_rate).run()


def run(config):
    """main.py 调用的唯一入口：按 config 运行策略（或读取已有持仓）、回测并生成报告。"""
    StrategyEvaluation(config).run()


if __name__ == "__main__":
    from main import load_config
    run(load_config("portfolio_20260905/config/config_trading_strategies.json"))
