from dataclasses import dataclass
@dataclass
class RegimeAwareAllocator:
    low_vol_optimizer: object | None = None
    medium_vol_optimizer: object | None = None
    high_vol_optimizer: object | None = None
    default_optimizer: object | None = None
    def allocate(self, returns, regimes):
        latest = str(regimes.dropna().iloc[-1])
        opt = self.default_optimizer
        if latest.startswith("low") and self.low_vol_optimizer is not None: opt = self.low_vol_optimizer
        elif latest.startswith("medium") and self.medium_vol_optimizer is not None: opt = self.medium_vol_optimizer
        elif latest.startswith("high") and self.high_vol_optimizer is not None: opt = self.high_vol_optimizer
        if opt is None: raise ValueError("no optimizer configured for latest regime")
        return opt.solve().weights
