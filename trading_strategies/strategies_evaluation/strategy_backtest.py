import os

import numpy as np
import pandas as pd

from factors.evaluation.single_factor_evaluation_v1 import SingleFactorEvaluation


class StrategyBacktest:
    LABELS = {"return": "理论", "net_return": "扣费", "slippage_return": "扣费+滑点"}
    DETAIL_COLUMNS = ["date", "code", "name", "target", "previous_weight", "weight", "buy_price", "sell_price", "stock_return", "contribution", "fee", "slippage", "up_limit", "down_limit", "suspended"]

    def __init__(self, holdings, backtest_data_dir, limit_price_dir, output_dir, fee_rate):
        self.holdings = holdings
        self.backtest_data_dir = backtest_data_dir
        self.limit_price_dir = limit_price_dir
        self.strategy_name = holdings["strategy"].iloc[0]
        self.output_dir = os.path.join(output_dir, self.strategy_name)
        self.fee_rate = fee_rate

    @staticmethod
    def _metrics(result, column):
        """计算区间收益、年化收益、年化波动、Sharpe、Sortino、最大回撤及其起止日、Calmar、日胜率、月胜率、持仓日占比、日均换手、最大单日收益 / 亏损、最长连亏天数和交易日数。"""
        nav = (1 + result[column]).cumprod()
        annual_return = nav.iloc[-1] ** (252 / len(nav)) - 1
        drawdown = nav / nav.cummax().clip(lower=1) - 1
        max_drawdown = drawdown.min()
        # 回撤结束日为回撤最深的一天，开始日为此前最后一个创新高（回撤为 0）的交易日，此前从未创新高时取区间首日。
        peaks = drawdown.loc[:drawdown.idxmin()]
        drawdown_start = peaks.index[peaks.eq(0)][-1] if peaks.eq(0).any() else result.index[0]
        # 月胜率只统计有持仓的月份；连亏天数按收益为负的连续交易日计。
        months = result.groupby(result.index.str[:6])["exposure"].sum().to_frame()
        months["return"] = (1 + result[column]).groupby(result.index.str[:6]).prod() - 1
        losing = result[column].lt(0)
        return {"区间收益": nav.iloc[-1] - 1, "年化收益": annual_return, "年化波动": result[column].std() * 252 ** 0.5, "Sharpe": result[column].mean() / result[column].std() * 252 ** 0.5,
                "Sortino": result[column].mean() / result[column].clip(upper=0).pow(2).mean() ** 0.5 * 252 ** 0.5, "最大回撤": max_drawdown, "最大回撤开始": drawdown_start, "最大回撤结束": drawdown.idxmin(),
                "Calmar": annual_return / -max_drawdown, "日胜率": result.loc[result["exposure"].gt(0), column].gt(0).mean(), "月胜率": months.loc[months["exposure"].gt(0), "return"].gt(0).mean(),
                "持仓日占比": result["exposure"].gt(0).mean(), "日均换手": result["turnover"].mean(), "最大单日收益": result[column].max(), "最大单日亏损": result[column].min(),
                "最长连亏天数": int(losing.groupby((~losing).cumsum()).cumsum().max()), "交易日数": len(result)}

    def run(self):
        """按持仓清单在 trade_date 10 点等权买入、持有到下一交易日 10 点，换手（买卖双边）乘 fee_rate 扣费；尚无收益的下一交易日不参与评估。保存并返回曲线、指标和逐股明细。"""
        prices = SingleFactorEvaluation.load_prices(self.backtest_data_dir)
        returns = (prices.shift(-1) / prices - 1).iloc[:-1]
        returns = returns.loc[self.holdings["trade_date"].min():self.holdings["trade_date"].max()]
        # 权重矩阵只保留清单中出现过的股票（含 hold 始终为 False 的），其余股票权重恒为 0，不影响结果。
        held = self.holdings.loc[self.holdings["hold"]]
        weights = pd.crosstab(held["trade_date"], held["code"])
        weights = weights.reindex(index=returns.index, columns=sorted(self.holdings["code"].unique()), fill_value=0).rename_axis(index="date", columns="code")
        returns = returns.reindex(columns=weights.columns)

        # 10 点价格（还原为未复权价）达到涨停价时不能买入或加仓，达到跌停价时不能卖出或减仓；unlimited 为 1 时无涨跌停限制。
        limits = pd.read_parquet(self.backtest_data_dir, columns=["code", "date", "close", "high", "low", "adj_factor", "is_suspended"])
        limits = limits.merge(pd.read_parquet(os.path.join(self.limit_price_dir, "limit_price.parquet")), on=["code", "date"])
        # 滑点：买入按 10 点最高价、卖出按 10 点最低价成交，相对 10 点收盘价损失的成交金额比例；需在 close 还原为未复权价之前计算。
        limits["buy_slippage"] = 1 - limits["close"] / limits["high"]
        limits["sell_slippage"] = 1 - limits["low"] / limits["close"]
        limits["close"] = (limits["close"] / limits["adj_factor"]).round(2)
        limits["up_limit"] = limits["close"].ge(limits["high_limit"]) & limits["unlimited"].eq(0)
        limits["down_limit"] = limits["close"].le(limits["low_limit"]) & limits["unlimited"].eq(0)
        up_limit = limits.pivot(index="date", columns="code", values="up_limit").reindex(index=returns.index, columns=returns.columns).eq(True)
        down_limit = limits.pivot(index="date", columns="code", values="down_limit").reindex(index=returns.index, columns=returns.columns).eq(True)
        suspended = limits.pivot(index="date", columns="code", values="is_suspended").reindex(index=returns.index, columns=returns.columns).eq(True)
        buy_slippage = limits.pivot(index="date", columns="code", values="buy_slippage").reindex(index=returns.index, columns=returns.columns).fillna(0)
        sell_slippage = limits.pivot(index="date", columns="code", values="sell_slippage").reindex(index=returns.index, columns=returns.columns).fillna(0)

        # 逐日调仓：跌停的原持仓保持原权重，其余资金等分给未被锁定的目标股票，涨停股票的权重不超过前一日；买不进的部分持有现金。
        target = weights.to_numpy() > 0
        actual = np.zeros(target.shape)
        previous = np.zeros(target.shape[1])
        for day in range(len(target)):
            locked = (previous > 0) & down_limit.iloc[day].to_numpy()
            actual[day] = np.where(target[day] & ~locked, (1 - previous[locked].sum()) / max((target[day] & ~locked).sum(), 1), 0)
            actual[day] = np.where(up_limit.iloc[day].to_numpy(), np.minimum(actual[day], previous), actual[day])
            actual[day] = np.where(locked, previous, actual[day])
            previous = actual[day]
        weights = pd.DataFrame(actual, index=weights.index, columns=weights.columns)
        weight_change = weights.diff().fillna(weights)

        # 曲线：无法买入（无价格）或停牌的股票收益记 0，相当于该部分持有现金；加仓部分按买入滑点、减仓部分按卖出滑点扣除。
        curve = pd.DataFrame(index=returns.index)
        curve["exposure"] = weights.sum(axis=1)
        curve["turnover"] = weight_change.abs().sum(axis=1)
        curve["return"] = (weights * returns.fillna(0)).sum(axis=1)
        curve["net_return"] = curve["return"] - curve["turnover"] * self.fee_rate
        curve["slippage_return"] = curve["net_return"] - (weight_change.clip(lower=0) * buy_slippage + weight_change.clip(upper=0).abs() * sell_slippage).sum(axis=1)
        for column in self.LABELS:
            curve[f"{column}_nav"] = (1 + curve[column]).cumprod()
            curve[f"{column}_drawdown"] = curve[f"{column}_nav"] / curve[f"{column}_nav"].cummax().clip(lower=1) - 1

        # 绩效表：全区间和每个年份分别按三种口径计算，回撤每年重置。
        rows = []
        for column in self.LABELS:
            rows.append({"区间": "全区间", "口径": self.LABELS[column], **self._metrics(curve, column)})
            for year, year_curve in curve.groupby(curve.index.str[:4]):
                rows.append({"区间": year, "口径": self.LABELS[column], **self._metrics(year_curve, column)})
        metrics = pd.DataFrame(rows)

        # 逐股明细：宽表 stack 成 date × code，只保留当日在清单中、当日有权重或前一日有权重的股票（后者保证清仓日的卖出滑点不丢）。
        details = pd.crosstab(self.holdings["trade_date"], self.holdings["code"]).reindex(index=returns.index, columns=weights.columns, fill_value=0)
        details = details.gt(0).stack().rename("target").to_frame()
        details["previous_weight"] = weights.shift(1, fill_value=0).stack()
        details["weight"] = weights.stack()
        details = details.loc[details["target"] | details["weight"].gt(0) | details["previous_weight"].gt(0)]
        details["buy_price"] = prices.reindex(index=returns.index, columns=weights.columns).stack()
        details["sell_price"] = prices.shift(-1).reindex(index=returns.index, columns=weights.columns).stack()
        details["stock_return"] = returns.stack()
        details["suspended"] = suspended.stack() | details["stock_return"].isna()
        details["stock_return"] = details["stock_return"].fillna(0)
        details["contribution"] = details["weight"] * details["stock_return"]
        details["fee"] = (details["weight"] - details["previous_weight"]).abs() * self.fee_rate
        details["slippage"] = (details["weight"] - details["previous_weight"]).clip(lower=0) * buy_slippage.stack() + (details["weight"] - details["previous_weight"]).clip(upper=0).abs() * sell_slippage.stack()
        details["up_limit"] = up_limit.stack()
        details["down_limit"] = down_limit.stack()
        details = details.reset_index()
        details["name"] = details["code"].map(self.holdings.drop_duplicates("code", keep="last").set_index("code")["name"])
        details = details[self.DETAIL_COLUMNS]

        os.makedirs(self.output_dir, exist_ok=True)
        curve.to_csv(os.path.join(self.output_dir, f"{self.strategy_name}Curve.csv"), index_label="date", encoding="utf-8-sig")
        metrics.to_csv(os.path.join(self.output_dir, f"{self.strategy_name}Metrics.csv"), index=False, encoding="utf-8-sig")
        details.to_csv(os.path.join(self.output_dir, f"{self.strategy_name}Details.csv"), index=False, encoding="utf-8-sig")
        return curve, metrics, details
