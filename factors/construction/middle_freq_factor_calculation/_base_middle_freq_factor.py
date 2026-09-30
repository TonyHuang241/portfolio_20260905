class _BaseMiddleFreqFactor:
    """中频因子基类：接收含 code、date 和价格字段的多股票历史日线。"""

    lookback_days = 0

    def __init__(self, daily, **kwargs):
        if daily is None:
            raise ValueError("daily cannot be None")
        self.daily = daily

    def calculate(self):
        """批量返回 code、date 和以因子类名命名的因子值列；勿修改共享的 daily。"""
        raise NotImplementedError("calculate is not defined")
