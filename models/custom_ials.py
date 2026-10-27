"""ALS target combines absolute rating and user-centered rating."""

from models.ials import IALS


class CustomIALS(IALS):
    def __init__(self, beta=0.5, **kwargs):
        super().__init__(**kwargs)
        self.beta = beta

    def targets(self, frame):
        means = frame.groupby("user_idx").rating.transform("mean").to_numpy()
        ratings = frame.rating.to_numpy()
        return self.beta * (ratings - 3) / 2 + (1 - self.beta) * (ratings - means) / 4
