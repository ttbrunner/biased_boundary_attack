from abc import ABC, abstractmethod
import numpy as np


# Distance measure: lower number = higher similarity.
# By default, accepts inputs in range [.0, .1]
class DistanceMeasure(ABC):

    # Calc distance between a and b.
    @abstractmethod
    def calc(self, a, b):
        pass

    # Subclasses may choose to override this for custom batching.
    def calc_batch(self, a, b):
        assert a.dtype == b.dtype
        assert a.shape == b.shape
        arr = np.empty(a.shape[1:], dtype=a.dtype)
        for i in range(a.shape[0]):
            arr[i, ...] = self.calc(a[i, ...], b[i, ...])

    # Make sure this distance measure is 255-based. If not, convert.
    def to_range_255(self):
        if isinstance(self, DistanceWrapper255):
            return self
        else:
            return DistanceWrapper255(self)

    # Make sure this distance measure is 0/1-based. If not, convert.
    def to_range_01(self):
        if isinstance(self, DistanceWrapper255):
            return self.wrapped_measure
        else:
            return self


# Wrapper for 255-based inputs
class DistanceWrapper255(DistanceMeasure):
    def __init__(self, wrapped_measure):
        assert not isinstance(wrapped_measure, DistanceWrapper255), "Attempted to 255-wrap a distance measure twice."
        self.wrapped_measure = wrapped_measure

    def calc(self, a, b):
        return self.wrapped_measure.calc(a / 255., b / 255.)

    def calc_batch(self, a, b):
        return self.wrapped_measure.calc_batch(a / 255., b / 255.)


class DistLInf(DistanceMeasure):
    def calc(self, a, b):
        assert a.shape == b.shape
        diff = a - b
        return np.max(np.abs(diff))


class DistL2(DistanceMeasure):
    def calc(self, a, b):
        assert a.shape == b.shape
        diff = a - b
        return np.linalg.norm(diff)


class DistMSE(DistanceMeasure):
    def calc(self, a, b):
        assert a.shape == b.shape
        diff = a - b
        return np.sum(np.square(diff)) / np.prod(a.shape)
