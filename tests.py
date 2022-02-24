import constants
import gen

import numpy as np

assert gen.collide(np.asarray((0., 0.)), np.asarray((0.5, 0.0)))
assert not gen.collide(np.asarray((0., 0.)), np.asarray((1., 1.)))
