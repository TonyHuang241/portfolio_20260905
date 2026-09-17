class _BaseHighFreqFactor:
    """Base class shared by all daily high-frequency factors."""

    def __init__(self, minutes, **kwargs):
        if minutes is None:
            raise ValueError("minutes cannot be None")
        self.minutes = minutes
        self.trade_date = minutes["date"].iloc[0] if not minutes.empty else None

    def calculate(self):
        raise NotImplementedError("calculate is not defined")
