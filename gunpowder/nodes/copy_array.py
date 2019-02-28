import logging
import numpy as np

from .batch_filter import BatchFilter
from gunpowder.array import Array

logger = logging.getLogger(__name__)

class CopyArray(BatchFilter):
    """Add a copy of an Array so that it can be processed in a different way
    Args:
        src_key (:class:`ArrayKey`):
            The array to copy.
        tgt_key (:class:`ArrayKey`):
            The target array.
        """
    def __init__(self, src_key, tgt_key):
        self.src_key = src_key
        self.tgt_key = tgt_key

    def setup(self):
        spec = self.spec[self.src_key].copy()
        self.provides(self.tgt_key, spec)

    def process(self, batch, request):
        if self.tgt_key not in request:
            return

        spec = self.spec[self.tgt_key].copy()
        spec.roi = request[self.tgt_key].roi
        batch.arrays[self.tgt_key] = Array(np.copy(batch.arrays[self.src_key].data), spec)