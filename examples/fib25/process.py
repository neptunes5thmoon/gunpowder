import sys
import os

import copy
import numpy as np

from gunpowder import *
from gunpowder.caffe import *
from gunpowder.contrib import ZeroOutConstSections

def get_spec(source, arraykey):
    source_fake = copy.deepcopy(source)
    with build(source_fake):
        spec = source_fake.spec[arraykey]
        return spec

def predict_affinities(gpu):

    set_verbose(False)

    # the network architecture
    prototxt = 'net.prototxt'

    # the learned weights (example at iteration 90000)
    iteration=90000
    weights  = 'net_iter_%d.caffemodel' %iteration

    # where to write output
    h5_targetfile='./processed_trvol-250-1_%d.hdf' %iteration

    # input and output sizes of the network (needed to formulate Chunk requests 
    # of the correct size later)
    voxel_size = Coordinate((8,)*3)
    input_size = Coordinate((196,)*3) * voxel_size
    output_size = Coordinate((92,)*3) * voxel_size

    # the size of the receptive field of the network
    context = (input_size - output_size)/2

    # add array keys
    ArrayKey('RAW')
    ArrayKey('PRED_AFFINITIES')

    # a chunk request that matches the dimensions of the network, will be used 
    # to chunk the whole array into batches of this size
    chunk_request = BatchRequest()
    chunk_request.add(ArrayKeys.RAW, input_size)
    chunk_request.add(ArrayKeys.PRED_AFFINITIES, output_size)
    chunk_request[ArrayKeys.RAW].voxel_size = voxel_size
    chunk_request[ArrayKeys.PRED_AFFINITIES].voxel_size = voxel_size

    # where to find the intensities
    source = Hdf5Source(
            'trvol-250-1.hdf',
            datasets={ ArrayKeys.RAW: 'volumes/raw'}
    )

    raw_spec = get_spec(source, ArrayKeys.RAW)
    pred_vol_spec = ArraySpec(roi=raw_spec.roi, voxel_size=raw_spec.voxel_size, dtype=np.float32)

    # the prediction pipeline:
    process_pipeline = (
            source +

            # ensure RAW is in float in [0,1]
            Normalize(ArrayKeys.RAW) +

            # zero-pad provided RAW to be able to draw batches close to the 
            # boundary of the available data
            Pad({ ArrayKeys.RAW: Coordinate(100, 100, 100) * voxel_size}) +

            # ensure RAW is in [-1,1]
            IntensityScaleShift(ArrayKeys.RAW, 2, -1) +
            ZeroOutConstSections(ArrayKeys.RAW) +

            # predict affinities
            Predict(prototxt,
                    weights,
                    inputs={'data': ArrayKeys.RAW},
                    outputs={'aff_pred': ArrayKeys.PRED_AFFINITIES},
                    array_specs={ArrayKeys.PRED_AFFINITIES: pred_vol_spec},
                    use_gpu=gpu) +

            # add useful profiling stats to identify bottlenecks
            PrintProfilingStats() +

            # write prediction to file
            Hdf5Write(dataset_names={ArrayKeys.RAW: 'volumes/raw',
                                     ArrayKeys.PRED_AFFINITIES: 'volumes/pred_affinitites'},
                      output_dir=os.path.dirname(h5_targetfile),
                      output_filename=os.path.basename(h5_targetfile),
                      dataset_dtypes={ArrayKeys.RAW: np.float32,
                                      ArrayKeys.PRED_AFFINITIES: np.float32}) +
            Scan(chunk_request)
    )

    with build(process_pipeline) as p:
        # when using scan node, actual requests are built within pipeline - so here we just use an empty request
        dummy_request = BatchRequest()
        p.request_batch(dummy_request)

if __name__ == "__main__":
    gpu = int(sys.argv[1])
    predict_affinities(gpu)
