import os

import numpy as np
import pandas as pd


class StrategyBacktest:
    LABELS = {"return": "理论", "net_return": "扣费", "slippage_return": "扣费+滑点"}
    DETAIL_COLUMNS = ["date", "code", "name", "target", "previous_shares", "shares", "buy_price", "sell_price", "stock_return", "pnl", "fee", "slippage", "up_limit", "down_limit", "suspended"]

    def __init__(self, holdings, backtest_data_dir, limit_price_dir, output_dir, fee_rate, min_fee, initial_cash):
        self.holdings = holdings
        self.backtest_data_dir = backtest_data_dir
        self.limit_price_dir = limit_price_dir
        self.strategy_name = holdings["strategy"].iloc[0]
        self.output_dir = os.path.join(output_dir, self.strategy_name)
        self.fee_rate = fee_rate
        self.min_fee = min_fee
        self.initial_cash = initial_cash

    @staticmethod
    def _metrics(result, column):
        """计算区间盈亏（元）、区间收益、年化收益、年化波动、Sharpe、Sortino、最大回撤及其起止日、Calmar、日胜率、月胜率、持仓日占比、日均换手、最大单日收益 / 亏损、最长连亏天数和交易日数。"""
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
        return {"区间盈亏": result[f"{column}_pnl"].sum(), "区间收益": nav.iloc[-1] - 1, "年化收益": annual_return, "年化波动": result[column].std() * 252 ** 0.5, "Sharpe": result[column].mean() / result[column].std() * 252 ** 0.5,
                "Sortino": result[column].mean() / result[column].clip(upper=0).pow(2).mean() ** 0.5 * 252 ** 0.5, "最大回撤": max_drawdown, "最大回撤开始": drawdown_start, "最大回撤结束": drawdown.idxmin(),
                "Calmar": annual_return / -max_drawdown, "日胜率": result.loc[result["exposure"].gt(0), column].gt(0).mean(), "月胜率": months.loc[months["exposure"].gt(0), "return"].gt(0).mean(),
                "持仓日占比": result["exposure"].gt(0).mean(), "日均换手": result["turnover"].mean(), "最大单日收益": result[column].max(), "最大单日亏损": result[column].min(),
                "最长连亏天数": int(losing.groupby((~losing).cumsum()).cumsum().max()), "交易日数": len(result)}

    def run(self):
        """按持仓清单在 trade_date 10 点用未复权价等权买入整 100 股、持有到下一交易日 10 点，成交金额（买卖双边）乘 fee_rate 扣费、每笔不足 min_fee 时按 min_fee 收；尚无收益的下一交易日不参与评估。保存并返回曲线、指标和逐股明细，金额单位为元。"""
        # 10 点行情只保留清单中出现过的股票（含 hold 始终为 False 的），价格除以复权因子还原为未复权价。
        data = pd.read_parquet(self.backtest_data_dir, columns=["code", "date", "close", "high", "low", "adj_factor", "is_suspended"])
        data = data.loc[data["code"].isin(self.holdings["code"]) & data["date"].ge(self.holdings["trade_date"].min())]
        data = data.merge(pd.read_parquet(os.path.join(self.limit_price_dir, "limit_price.parquet")), on=["code", "date"], how="left")
        for column in ["close", "high", "low"]:
            data[column] = (data[column] / data["adj_factor"]).round(2)
        # 10 点价格达到涨停价时不能买入或加仓，达到跌停价时不能卖出或减仓；unlimited 为 1 时无涨跌停限制。
        data["up_limit"] = data["close"].ge(data["high_limit"]) & data["unlimited"].eq(0)
        data["down_limit"] = data["close"].le(data["low_limit"]) & data["unlimited"].eq(0)
        data = data.pivot(index="date", columns="code", values=["close", "high", "low", "adj_factor", "is_suspended", "up_limit", "down_limit"])
        # 数值列和布尔列一起 pivot 后都是 object 类型，统一转成浮点数（布尔标记为 1 / 0，缺失为 NaN）。
        data = data.astype(float)
        # 评估区间到清单最后一个 trade_date 为止；行情最后一天没有下一交易日价格，不参与评估。
        dates = data.index[:-1]
        dates = dates[dates <= self.holdings["trade_date"].max()]

        # 没有行情（未上市、退市）的日期视为停牌，不能买卖，价格沿用最近一次的 10 点价格；买入按 10 点最高价、卖出按 10 点最低价成交。
        price = data["close"].ffill().bfill()
        suspended = data["is_suspended"].eq(True) | data["close"].isna()
        suspended = suspended.loc[dates]
        up_limit = data["up_limit"].eq(True).loc[dates]
        down_limit = data["down_limit"].eq(True).loc[dates]
        high = data["high"].fillna(price).loc[dates]
        low = data["low"].fillna(price).loc[dates]
        # 持仓数量每日按复权因子的变化调整（送转、分红都折算成股数）；next_price 为下一交易日 10 点价格按当日股数口径折算后的价格。
        adjust = data["adj_factor"].ffill()
        adjust = (adjust / adjust.shift(1)).fillna(1)
        next_price = (price * adjust).shift(-1).loc[dates]
        price = price.loc[dates]
        adjust = adjust.loc[dates]

        held = self.holdings.loc[self.holdings["hold"]]
        weights = pd.crosstab(held["trade_date"], held["code"])
        weights = weights.reindex(index=dates, columns=price.columns, fill_value=0).rename_axis(index="date", columns="code")

        # 逐日调仓：跌停或停牌的原持仓保持不动，其余资产（现金 + 可卖持仓按 10 点最低价扣费后的变现金额）等分给未被锁定的目标股票，每只先留出一笔最低佣金，再按 10 点最高价加费用向下取整到 100 股；
        # 加仓只能买整 100 股，涨停或停牌的股票股数不超过调仓前；买不进和取整剩下的部分持有现金，按变现金额分钱保证现金不会透支。
        # 每只股票每天的买入或卖出算一笔，费用 = 成交金额 × fee_rate，不足 min_fee 时按 min_fee 收；调整后的股数保留 6 位小数，避免浮点误差被当成一笔成交。
        buy_price = high.to_numpy()
        sell_price = low.to_numpy()
        target = weights.to_numpy() > 0
        actual = np.zeros(target.shape)
        previous = np.zeros(target.shape[1])
        cash = self.initial_cash
        for day in range(len(target)):
            previous = np.round(previous * adjust.iloc[day].to_numpy(), 6)
            locked = (previous > 0) & (down_limit.iloc[day].to_numpy() | suspended.iloc[day].to_numpy())
            sellable = (previous > 0) & ~locked
            budget = (cash + (previous * sell_price[day] - np.maximum(previous * sell_price[day] * self.fee_rate, self.min_fee))[sellable].sum()) / max((target[day] & ~locked).sum(), 1)
            actual[day] = np.where(target[day] & ~locked, np.floor(max(budget - self.min_fee, 0) / (buy_price[day] * (1 + self.fee_rate)) / 100) * 100, 0)
            actual[day] = np.where(actual[day] > previous, previous + np.floor((actual[day] - previous) / 100) * 100, actual[day])
            actual[day] = np.where(up_limit.iloc[day].to_numpy() | suspended.iloc[day].to_numpy(), np.minimum(actual[day], previous), actual[day])
            actual[day] = np.where(locked, previous, actual[day])
            # amount 为各股票的成交金额，买入为正、卖出为负。
            amount = (actual[day] - previous) * np.where(actual[day] > previous, buy_price[day], sell_price[day])
            cash = cash - amount.sum() - np.maximum(np.abs(amount) * self.fee_rate, self.min_fee)[amount != 0].sum()
            previous = actual[day]
        shares = pd.DataFrame(actual, index=weights.index, columns=weights.columns)
        previous_shares = (shares.shift(1, fill_value=0) * adjust).round(6)
        trade = shares - previous_shares
        # 费用按成交金额计，每笔不足 min_fee 时按 min_fee 收；滑点为成交价相对 10 点价格多付（买入）或少收（卖出）的金额。
        fee = (trade.clip(lower=0) * high + trade.clip(upper=0).abs() * low) * self.fee_rate
        fee = fee.clip(lower=self.min_fee).where(trade.ne(0), 0)
        slippage = trade.clip(lower=0) * (high - price) + trade.clip(upper=0).abs() * (price - low)

        # 曲线：现金和持仓市值为当日 10 点调仓后的金额；盈亏为持仓到下一交易日 10 点的价差，三种口径依次扣除费用和滑点；资产 = 初始现金 + 累计盈亏，日收益率 = 当日盈亏 / 前一日资产。
        curve = pd.DataFrame(index=dates)
        curve["cash"] = self.initial_cash - (trade * price + fee + slippage).sum(axis=1).cumsum()
        curve["position"] = (shares * price).sum(axis=1)
        curve["exposure"] = curve["position"] / (curve["cash"] + curve["position"])
        curve["turnover"] = (trade.abs() * price).sum(axis=1) / (curve["cash"] + curve["position"])
        curve["return_pnl"] = (shares * (next_price - price)).sum(axis=1)
        curve["net_return_pnl"] = curve["return_pnl"] - fee.sum(axis=1)
        curve["slippage_return_pnl"] = curve["net_return_pnl"] - slippage.sum(axis=1)
        for column in self.LABELS:
            curve[f"{column}_asset"] = self.initial_cash + curve[f"{column}_pnl"].cumsum()
            curve[column] = curve[f"{column}_pnl"] / (curve[f"{column}_asset"] - curve[f"{column}_pnl"])
            curve[f"{column}_drawdown"] = curve[f"{column}_asset"] / curve[f"{column}_asset"].cummax().clip(lower=self.initial_cash) - 1

        # 绩效表：全区间和每个年份分别按三种口径计算，回撤每年重置。
        rows = []
        for column in self.LABELS:
            rows.append({"区间": "全区间", "口径": self.LABELS[column], **self._metrics(curve, column)})
            for year, year_curve in curve.groupby(curve.index.str[:4]):
                rows.append({"区间": year, "口径": self.LABELS[column], **self._metrics(year_curve, column)})
        metrics = pd.DataFrame(rows)

        # 逐股明细：宽表 stack 成 date × code，只保留当日在清单中、当日有持仓或调仓前有持仓的股票（后者保证清仓日的卖出费用和滑点不丢）。
        details = pd.crosstab(self.holdings["trade_date"], self.holdings["code"]).reindex(index=dates, columns=shares.columns, fill_value=0)
        details = details.gt(0).stack().rename("target").to_frame()
        details["previous_shares"] = previous_shares.stack()
        details["shares"] = shares.stack()
        details = details.loc[details["target"] | details["shares"].gt(0) | details["previous_shares"].gt(0)]
        details["buy_price"] = price.stack()
        details["sell_price"] = next_price.stack()
        details["stock_return"] = details["sell_price"] / details["buy_price"] - 1
        details["pnl"] = details["shares"] * (details["sell_price"] - details["buy_price"])
        details["fee"] = fee.stack()
        details["slippage"] = slippage.stack()
        details["up_limit"] = up_limit.stack()
        details["down_limit"] = down_limit.stack()
        details["suspended"] = suspended.stack()
        details = details.reset_index()
        details["name"] = details["code"].map(self.holdings.drop_duplicates("code", keep="last").set_index("code")["name"])
        details = details[self.DETAIL_COLUMNS]

        os.makedirs(self.output_dir, exist_ok=True)
        curve.to_csv(os.path.join(self.output_dir, f"{self.strategy_name}Curve.csv"), index_label="date", encoding="utf-8-sig")
        metrics.to_csv(os.path.join(self.output_dir, f"{self.strategy_name}Metrics.csv"), index=False, encoding="utf-8-sig")
        details.to_csv(os.path.join(self.output_dir, f"{self.strategy_name}Details.csv"), index=False, encoding="utf-8-sig")
        return curve, metrics, details
