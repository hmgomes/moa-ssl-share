import typing as t
import numpy as np
from river.base.typing import Stream, ClfTarget

Instance = t.Tuple[dict, t.Any]
SemiSupervisedLabel = t.Optional[ClfTarget]


class SemiSupervisedStream(Stream):
    """A stream that returns only some instances with a label. The remaining
    instances are returned without a label.

    Consistency across implementations is guaranteed using the MT19937
    pseudorandom number generator. The seed is used to initialize the
    generator. Following the warmup period, the generator is then used to
    generate a random number between 0 and 1 for each element in the stream.
    If the random number is less than the probability `p`, the element is
    returned with a label. Otherwise, the element is returned without a label.
    """

    def __init__(
        self,
        stream: Stream,
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
        self.delay = int(delay) if delay is not None else None
        self.iterator = self._generate_iterator()

    def __iter__(self) -> t.Iterator[t.Tuple[dict, SemiSupervisedLabel, ClfTarget]]:
        return self.iterator

    def _generate_iterator(self):
        # Each new stream resets the random number generator's state
        mt19937 = np.random.MT19937()
        mt19937._legacy_seeding(self.seed)
        rand = np.random.Generator(mt19937)

        # If delay is defined, we need to buffer the instances. The buffer
        # is a list of tuples. The first element is the index when
        # it should be emitted. The second element is the instance itself.
        _delay_buffer: t.List[t.Tuple[int, Instance]] = []

        self._i = 0
        for x, y in self.wrapped_stream:
            # If we have delayed instances, check if we should emit them
            while len(_delay_buffer) > 0 and _delay_buffer[0][0] == self._i:
                self._i += 1
                buf_x, buf_y = _delay_buffer.pop(0)[1]
                yield buf_x, buf_y, buf_y

            # If the warmup is not over, return the element
            if self._i < self.warmup_length:
                self._i += 1
                yield x, y, y
            # Otherwise, return the element with probability p
            elif rand.random(dtype=np.float64) >= self.label_p:
                self._i += 1
                yield x, None, y

                # Buffer the instance if delay is defined
                if self.delay is not None:
                    _delay_buffer.append((self._i + self.delay, (x, y)))
            else:
                self._i += 1
                yield x, y, y

    def __next__(self) -> t.Tuple[dict, SemiSupervisedLabel, ClfTarget]:
        return next(self.iterator)

    @property
    def is_warming_up(self) -> bool:
        return self.warmup_length > self._i
