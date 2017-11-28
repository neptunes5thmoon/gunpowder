from .batch_filter import BatchFilter
from gunpowder.volume_spec import VolumeSpec
from gunpowder.provider_spec import ProviderSpec
from gunpowder.batch import Batch
from gunpowder.volume import Volume
from gunpowder.coordinate import Coordinate
import numpy as np
import copy
class Fill(BatchFilter):

    def __init__(
            self, fill_types_and_values, volume_specs=None):

        self.fill_types_and_values = fill_types_and_values
        self.volume_specs = {} if volume_specs is None else volume_specs

    def setup(self):
        for fill_type in self.fill_types_and_values.keys():
            if fill_type in self.volume_specs:
                spec = copy.deepcopy(self.volume_specs[fill_type])

            else:
                spec = VolumeSpec()
                spec.voxel_size=Coordinate((1,1,1))

            self.provides(fill_type, spec)

    def prepare(self, request):

        for volume_type in self.fill_types_and_values.keys():
            if volume_type in request:
                del request[volume_type]

    def process(self, batch, request):
        #upstream_request = copy.deepcopy(request)

        #for upstream_provider in self.get_upstream_providers():
        #    batch.update(upstream_provider.request_batch(upstream_request))
        for (volume_type, request_spec) in request.volume_specs.items():
            if volume_type in self.fill_types_and_values.keys():
                voxel_size = self.spec[volume_type].voxel_size
                dataset_roi = request_spec.roi/voxel_size
                dataset_roi = dataset_roi - self.spec[volume_type].roi.get_offset()/voxel_size
                volume_spec = self.spec[volume_type].copy()
                volume_spec.roi = request_spec.roi

                batch.volumes[volume_type] = Volume(self.fill_types_and_values[volume_type]*np.ones(
                    dataset_roi.get_shape(),
                                                                                   dtype=volume_spec.dtype), volume_spec)

        # batch = Batch()
        # for (volume_type, request_spec) in request.volume_specs.items():
        #     voxel_size = self.spec[volume_type].voxel_size
        #     dataset_roi = request_spec.roi/voxel_size
        #     dataset_roi = dataset_roi - self.spec[volume_type].roi.get_offset()/voxel_size
        #     volume_spec = self.spec[volume_type].copy()
        #     volume_spec.roi = request_spec.roi
        #     batch.volumes[volume_type] = Volume(self.fill_types_and_values[volume_type]*np.ones(dataset_roi,
        #                                                                            dtype=volume_spec.dtype), volume_spec)
