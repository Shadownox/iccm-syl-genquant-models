import collections

import ccobra
import numpy as np


class Random(ccobra.CCobraModel):
    def __init__(self, name='Random'):
        super(Random, self).__init__(name, ['syllogistic-generalized'], ['single-choice'])

    def predict(self, item, **kwargs):
        idx = np.random.randint(len(item.choices))
        return item.choices[idx]
