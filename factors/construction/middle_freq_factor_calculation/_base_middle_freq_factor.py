class _BaseMiddleFreqFactor:
    """中频因子基类：接收含 code、date 和价格字段的多股票历史日线，以及以 code、date（公告日）为主键的季度财务数据（报告期见 year、quarter，流量变量为 *_ttm）。"""

    lookback_days = 0

    def __init__(self, daily, financial_data=None, **kwargs):
        if daily is None:
            raise ValueError("daily cannot be None")
        self.daily = daily
        self.financial_data = financial_data

    def calculate(self):
        """批量返回 code、date 和以因子类名命名的因子值列；勿修改共享的 daily 和 financial_data，使用财务数据时按 date 向前取最近一次公告，避免未来信息。"""
        raise NotImplementedError("calculate is not defined")
