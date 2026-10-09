class _BaseMiddleFreqFactor:
    """中频因子基类：接收含 code、date 和价格字段的多股票历史日线，以及以 code、report_date（报告期末）为主键的季度财务数据（公告日为 date，同日可能公告多期；流量变量为 *_ttm 和当季度值 *_q）。"""

    lookback_days = 0

    def __init__(self, daily, financial_data=None, **kwargs):
        if daily is None:
            raise ValueError("daily cannot be None")
        self.daily = daily
        self.financial_data = financial_data

    def calculate(self):
        """批量返回 code、date 和以因子类名命名的因子值列；勿修改共享的 daily 和 financial_data，使用财务数据时按 date 向前取最近一次公告、同日取 report_date 最新的一期，避免未来信息。"""
        raise NotImplementedError("calculate is not defined")
