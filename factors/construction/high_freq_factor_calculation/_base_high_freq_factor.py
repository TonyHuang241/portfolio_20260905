class _BaseHighFreqFactor:
    """高频因子基类：接收单个交易日多只股票的分钟行情，calculate() 返回 code、date 和以因子类名命名的一列因子值。"""

    def __init__(self, minutes, **kwargs):
        if minutes is None:
            raise ValueError("minutes cannot be None")
        self.minutes = minutes
        self.trade_date = minutes["date"].iloc[0] if not minutes.empty else None

    def calculate(self):
        raise NotImplementedError("calculate is not defined")
