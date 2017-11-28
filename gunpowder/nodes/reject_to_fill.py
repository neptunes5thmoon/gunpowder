import logging
import copy
import operator
import numpy as np

from .batch_provider import BatchProvider
from gunpowder.batch import Batch
from gunpowder.profiling import Timing
from gunpowder.volume import Volume, VolumeTypes

logger = logging.getLogger(__name__)

class RejectToFill(BatchProvider):

    def __init__(self, fill_types, accepted_case_node,
                 rejected_case_node, condition=None, mask_volume_type=VolumeTypes.GT_MASK):
        #self.min_masked = min_masked
        self.mask_volume_type = mask_volume_type
        self.fill_types = fill_types
        self.accepted_case_node = accepted_case_node
        self.rejected_case_node = rejected_case_node
        if condition is None:
            def f_condtion(x, value=0.):
                return x != value
            self.condition = f_condtion
        else:
            self.condition = condition

    def setup(self):
        # assert self.mask_volume_type in self.spec, "Reject can only be used if %s is provided"%self.mask_volume_type
        # self.upstream_provider = self.get_upstream_provider()
        #
        # spec = self.spec[self.mask_volume_type].copy()
        #
        # for volume_type in self.fill_types:
        #     self.provides(volume_type, spec)
        assert len(self.get_upstream_providers())>0

        common_spec = None

        for provider in self.get_upstream_providers():
            if common_spec is None:
                common_spec = copy.deepcopy(provider.spec)
            else:
                for identifier, spec in provider.spec.items():
                    if identifier not in common_spec:
                        del common_spec[identifier]
#
        for identifier, spec in common_spec.items():
            self.provides(identifier, spec)
        for provider_idx, provider in enumerate(self.get_upstream_providers()):
            if isinstance(provider.output, self.accepted_case_node):
                self.accepted_case_provider_idx = provider_idx
            elif isinstance(provider.output, self.rejected_case_node):
                self.rejected_case_provider_idx = provider_idx
            else:
                raise ValueError("Node RejectToFill got provider {0:} with output of type {1:} that is neither of the "
                                 "accepted ({"
                                 "2:}) nor the "
                                 "rejected ({3:})"
                                 "case node".format(provider, type(provider),self.accepted_case_node,
                                                    self.rejected_case_node))

    def provide(self, request):

        report_next_timeout = 10
        num_rejected = 0

        timing = Timing(self)
        timing.start()

        assert self.mask_volume_type in request, "Reject can only be used if a GT mask is requested"

        have_good_batch = False
        batch = self.get_upstream_providers()[self.rejected_case_provider_idx].request_batch(request)

        mask_ratio = batch.volumes[self.mask_volume_type].data.mean()

        have_good_batch = self.condition(mask_ratio)

        if have_good_batch:
            logger.debug(
                "good batch with mask ratio %f found at " % mask_ratio +
                str(batch.volumes[self.mask_volume_type].spec.roi))
            batch= self.get_upstream_providers()[self.accepted_case_provider_idx].request_batch(request)
        return batch
       # # while not have_good_batch:
       #
       #  #    batch = self.upstream_provider.request_batch(request)
       #      mask_ratio = batch.volumes[self.mask_volume_type].data.mean()
       #      have_good_batch = mask_ratio>=self.min_masked
       #
       #      if not have_good_batch:
       #
       #          logger.debug(
       #              "reject batch with mask ratio %f at "%mask_ratio +
       #              str(batch.volumes[self.mask_volume_type].spec.roi))
       #          num_rejected += 1
       #
       #          if timing.elapsed() > report_next_timeout:
       #
       #              logger.warning("rejected %d batches, been waiting for a good one since %ds"%(num_rejected, report_next_timeout))
       #              report_next_timeout *= 2


        timing.stop()
        batch.profiling_stats.add(timing)

        return batch
