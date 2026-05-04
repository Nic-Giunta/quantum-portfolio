import random

import numpy as np


def set_random_seed(seed: int | None) -> None:
    if seed is not None:
        random.seed(seed); np.random.seed(seed)
