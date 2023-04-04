import typing as t

import numpy as np

Instance = t.Tuple[dict, t.Any]


class SemiSupervisedStream(t.Iterable[Instance]):
    """A stream that returns only some instances with a label. The remaining
    instances are returned without a label.

    Consistency across implementations is guaranteed using the MT19937
    pseudorandom number generator. The seed is used to initialize the
    generator. Following the warmup period, the generator is then used to
    generate a random number between 0 and 1 for each element in the stream. 
    If the random number is less than the probability `p`, the element is 
    returned with a label. Otherwise, the element is returned without a label.
    """

    def __init__(self,
                 stream: t.Iterable[Instance],
                 label_p: float,
                 seed: int,
                 warmup: int = 0,
                 delay: t.Optional[int] = None,
            ):
        """A stream that returns instances with a label with probability p.

        :param stream: The underlying stream which is being wrapped.
        :param label_p: Probability of returning an element with a
            label. Must be in 0 <= p <= 1.
        :param seed: Seed for the random number generator.
        :param warmup: Number of instances to return with a label before
            switching to the semi-supervised setting.
        :param delay: When defined, instead of unlabeled instances we have 
            instances with delayed labels. The `delay` specifies the number
            of instances that must be seen before the instance is repeated
            with a label.
        """
        if not (0.0 <= label_p <= 1.0):
            raise ValueError("p must be in 0 <= p <= 1")

        self.wrapped_stream = stream
        self.label_p = float(label_p)
        self.seed = int(seed)
        self.warmup_length = int(warmup)
        self.delay = int(delay) if delay != None else None

    def __iter__(self):
        # Each new stream resets the random number generator's state
        mt19937 = np.random.MT19937()
        mt19937._legacy_seeding(self.seed)
        rand = np.random.Generator(mt19937)

        # If delay is defined, we need to buffer the instances. The buffer
        # is a list of tuples. The first element is the index when
        # it should be emitted. The second element is the instance itself.
        self._delay_buffer: t.List[t.Tuple[int, Instance]] = []

        i = 0
        for x, y in self.wrapped_stream:
            # If we have delayed instances, check if we should emit them
            while len(self._delay_buffer) > 0 and self._delay_buffer[0][0] == i:
                i += 1
                yield self._delay_buffer.pop(0)[1]

            # If the warmup is not over, return the element
            if i < self.warmup_length:
                i += 1
                yield x, y
            # Otherwise, return the element with probability p
            elif rand.random(dtype=np.float64) >= self.label_p:
                i += 1
                yield x, None

                # Buffer the instance if delay is defined
                if self.delay is not None:
                    self._delay_buffer.append((i + self.delay, (x, y)))
            else:
                i += 1
                yield x, y
