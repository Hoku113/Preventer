import math


def get_alpha(rate=30, cutoff=1):
    tau = 1 / (2 * math.pi * cutoff)
    te = 1 / rate
    return 1 / (1 + tau / te)

class OneEuroFilter:
    def __init__(self, freq=15, mincutoff=1, beta=1, dcutoff=1):
        self.freq = freq
        self.mincutoff = mincutoff
        self.beta = beta
        self.dcutoff = dcutoff
        self.x_previous = None
        self.dx = 0

    def __call__(self, x):
        dx_smoothed = self._calc_low_pass_filter(self.dx, get_alpha(self.freq, self.dcutoff))
        cutoff = self.mincutoff + self.beta * abs(dx_smoothed)
        x_filtered = self._calc_low_pass_filter(x, get_alpha(self.freq, cutoff))
        self.x_previous = x
        return x_filtered

    def _calc_low_pass_filter(self, x, alpha=0.5):
        if self.x_previous is None:
            self.x_previous = x
            return x
        x_filtered = alpha * x + (1 - alpha) * self.x_previous
        self.x_previous = x_filtered
        return x_filtered
