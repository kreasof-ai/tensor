"""AdamW template: unscale, reduce the global norm, clip, and update on Tensor."""

import math
import numpy as np


class AdamW:
    def __init__(self, library, parameters, *, loss_scale, learning_rate, clip):
        self.library = library
        self.parameters = parameters
        self.loss_scale = loss_scale
        self.learning_rate = learning_rate
        self.limit = clip
        self.steps = 0
        total = sum(math.ceil(p.weight.shape[0] / 1024) for p in parameters.values())
        self.parts = library.device.empty((total,))
        self.sums = library.device.empty((math.ceil(total / 1024),))
        self.clip = library.device.empty((2,))

    def step(self):
        run = self.library.run
        offset = 0
        for parameter in self.parameters.values():
            n = parameter.weight.shape[0]
            run(
                "sumsq",
                dict(n=n, total=self.parts.shape[0], scale=self.loss_scale),
                parameter.grad,
                self.parts,
                offset,
            )
            offset += math.ceil(n / 1024)
        run("sum_parts", dict(n=self.parts.shape[0]), self.parts, self.sums)
        run("clip", dict(n=self.sums.shape[0], limit=self.limit), self.sums, self.clip)
        # Check before changing weights, moments, or the optimizer's step counter.
        self.read_norm()
        self.steps += 1
        for parameter in self.parameters.values():
            run(
                "adamw",
                dict(
                    n=parameter.weight.shape[0],
                    scale=self.loss_scale,
                    decay=0.1 if len(parameter.shape) > 1 else 0.0,
                ),
                parameter.weight,
                parameter.half,
                parameter.grad,
                parameter.moment,
                parameter.variance,
                self.clip,
                self.learning_rate,
                1 - 0.9**self.steps,
                1 - 0.95**self.steps,
            )

    def read_norm(self):
        values = self.clip.to_numpy()
        if not np.isfinite(values).all():
            raise FloatingPointError(
                "non-finite gradients; optimizer update was skipped"
            )
        return float(values[1])
