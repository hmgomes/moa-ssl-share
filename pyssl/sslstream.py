import typing as t

import numpy as np


class SemiSupervisedStream(t.Iterable[t.Tuple[dict, t.Any]]):
    """A stream that returns only some instances with a label. The remaining
    instances are returned without a label.

    Consistency across implementations is guaranteed using the MT19937
    pseudorandom number generator. The seed is used to initialize the
    generator. Following the warmup period, the generator is then used to
    generate a random number between 0 and 1 for each element in the stream. 
    If the random number is less than the probability `p`, the element is 
    returned with a label. Otherwise, the element is returned without a label.
    """

    def __init__(self, p: float, seed: int, warmup_length: int, wrapped_stream: t.Iterable[t.Tuple[dict, t.Any]]):
        """A stream that returns elements with a label with probability p.

        :param p: Probability of returning an element with a label. Must be in 
            0 <= p <= 1.
        :param warmup_length: Number of elements to return with a label before
            switching to the semi-supervised setting.
        :param seed: Seed for the random number generator.
        :param wrapped_stream: The stream to wrap.
        """
        if not (0.0 <= p <= 1.0):
            raise ValueError("p must be in 0 <= p <= 1")

        self.p = p
        self.warmup_length = warmup_length
        self.wrapped_stream = wrapped_stream
        self.seed = seed

    def __iter__(self):
        # Each new stream resets the random number generator's state
        mt19937 = np.random.MT19937()
        mt19937._legacy_seeding(self.seed)
        rand = np.random.Generator(mt19937)

        for i, (x, y) in enumerate(self.wrapped_stream):
            # If the warmup is not over, return the element
            if i < self.warmup_length:
                yield x, y
            # Otherwise, return the element with probability p
            elif rand.random(dtype=np.float64) <= self.p:
                yield x, y
            else:
                yield x, None
