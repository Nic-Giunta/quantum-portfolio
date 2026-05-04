class PartialFillModel:
    def fill_ratio(self, requested_notional: float, adv: float, max_participation: float=0.1) -> float:
        return 1.0 if requested_notional <= 0 else min(1.0, adv*max_participation/requested_notional)
