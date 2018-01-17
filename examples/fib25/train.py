from __future__ import print_function
import sys
from gunpowder import *
from gunpowder.caffe import *
from gunpowder.contrib import PrepareMalis
import malis
import glob
import math
import numpy as np

# the training HDF files
samples = [
    'trvol-250-1.hdf',
    # add more here
]

# after how many iterations to switch from Euclidean loss to MALIS
phase_switch = 10000

def train_until(max_iteration, gpu):
    '''Resume training from the last stored network weights and train until ``max_iteration``.'''

    set_verbose(False)

    # get most recent training result
    solverstates = [ int(f.split('.')[0].split('_')[-1]) for f in glob.glob('net_iter_*.solverstate') ]
    if len(solverstates) > 0:
        trained_until = max(solverstates)
        print("Resuming training from iteration " + str(trained_until))
    else:
        trained_until = 0
        print("Starting fresh training")

    if trained_until < phase_switch < max_iteration:
        # phase switch lies in-between, split training into to parts
        train_until(phase_switch, gpu)
        trained_until = phase_switch

    if max_iteration <= phase_switch:
        phase = 'euclid'
    else:
        phase = 'malis'
    print("Traing until " + str(max_iteration) + " in phase " + phase)

    # setup training solver and network
    solver_parameters = SolverParameters()
    solver_parameters.train_net = 'net.prototxt'
    solver_parameters.base_lr = 0.5e-4
    solver_parameters.momentum = 0.95
    solver_parameters.momentum2 = 0.999
    solver_parameters.delta = 1e-8
    solver_parameters.weight_decay = 0.000005
    solver_parameters.lr_policy = 'inv'
    solver_parameters.gamma = 0.0001
    solver_parameters.power = 0.75
    solver_parameters.snapshot = 2000
    solver_parameters.snapshot_prefix = 'net'
    solver_parameters.type = 'Adam'
    if trained_until > 0:
        solver_parameters.resume_from = 'net_iter_' + str(trained_until) + '.solverstate'
    else:
        solver_parameters.resume_from = None
    solver_parameters.train_state.add_stage(phase)

    # input and output shapes of the network, needed to formulate matching batch 
    # requests
    voxel_size = Coordinate((8,)*3)
    input_shape = Coordinate((196,)*3)*voxel_size
    output_shape = Coordinate((92,)*3)*voxel_size

    # register array keys
    ArrayKey('RAW')
    ArrayKey('ALPHA_MASK')
    ArrayKey('GT_LABELS')
    ArrayKey('GT_MASK')
    ArrayKey('PRED_AFFINITIES')
    ArrayKey('GT_AFFINITIES')
    ArrayKey('GT_AFFINITIES_MASK')
    ArrayKey('GT_AFFINITIES_SCALE')
    ArrayKey('LOSS_GRADIENT')
    ArrayKey('MALIS_COMP_LABEL')

    # arrays to request for each batch
    request = BatchRequest()
    request.add(ArrayKeys.RAW, input_shape)
    request.add(ArrayKeys.GT_LABELS, output_shape)
    request.add(ArrayKeys.GT_MASK, output_shape)
    request.add(ArrayKeys.GT_AFFINITIES, output_shape)
    request.add(ArrayKeys.GT_AFFINITIES_MASK, output_shape)
    request.add(ArrayKeys.MALIS_COMP_LABEL, output_shape)

    if phase == 'euclid':
        request.add(ArrayKeys.GT_AFFINITIES_SCALE, output_shape)

    # create a tuple of data sources, one for each HDF file
    data_sources = tuple(

        # provide arrays from the given HDF datasets
        Hdf5Source(
            sample,
            datasets = {
                ArrayKeys.RAW: 'volumes/raw',
                ArrayKeys.GT_LABELS: 'volumes/labels/neuron_ids',
                ArrayKeys.GT_MASK: 'volumes/labels/mask',
            },
            array_specs={
                ArrayKeys.GT_MASK: ArraySpec(interpolatable=False)
            }
        ) +

        # ensure RAW is in float in [0,1]
        Normalize(ArrayKeys.RAW) +

        # zero-pad provided RAW and GT_MASK to be able to draw batches close to 
        # the boundary of the available data
        Pad(
            {
                ArrayKeys.RAW: Coordinate((100, 100, 100))*voxel_size,
                ArrayKeys.GT_LABELS: Coordinate((100, 100, 100))*voxel_size,
                ArrayKeys.GT_MASK: Coordinate((100, 100, 100)*voxel_size)
            }
        ) +

        # chose a random location inside the provided arrays
        RandomLocation() +

        # reject batches wich do contain less than 50% labelled data
        Reject(ArrayKeys.GT_MASK)

        for sample in samples
    )

    # attach data sources to training pipeline
    train_pipeline = (

        data_sources +

        # randomly select any of the data sources
        RandomProvider() +

        # elastically deform and rotate
        ElasticAugment([40,40,40], [2,2,2], [0,math.pi/2.0], prob_slip=0.01, max_misalign=1, subsample=8) +

        # randomly mirror and transpose
        SimpleAugment() +

        # grow a 0-boundary between labelled objects
        GrowBoundary(ArrayKeys.GT_LABELS, ArrayKeys.GT_MASK, steps=4) +

        # relabel connected label components inside the batch
        SplitAndRenumberSegmentationLabels(ArrayKeys.GT_LABELS) +

        # compute ground-truth affinities from labels
        AddGtAffinities([[-1,0,0], [0, -1, 0], [0, 0, -1]],
                        gt_labels=ArrayKeys.GT_LABELS,
                        gt_affinities=ArrayKeys.GT_AFFINITIES,
                        gt_labels_mask=ArrayKeys.GT_MASK,
                        gt_affinities_mask=ArrayKeys.GT_AFFINITIES_MASK) +
        PrepareMalis(
            labels_array_key=ArrayKeys.GT_LABELS,
            malis_comp_array_key=ArrayKeys.MALIS_COMP_LABEL
        ) +

        # add a GT_AFFINITIES_SCALE array to balance positive and negative classes for
        # Euclidean training
        BalanceLabels(ArrayKeys.GT_AFFINITIES,
                      ArrayKeys.GT_AFFINITIES_SCALE,
                      ArrayKeys.GT_AFFINITIES_MASK) +

        # randomly scale and shift intensities
        IntensityAugment(ArrayKeys.RAW, 0.9, 1.1, -0.1, 0.1) +

        # ensure RAW is in [-1,1]
        IntensityScaleShift(ArrayKeys.RAW, 2,-1) +
        ZeroOutConstSections(ArrayKeys.RAW) +

        # use 10 workers to pre-cache batches of the above pipeline
        PreCache(
            cache_size=40,
            num_workers=10) +

        # perform one training iteration
        Train(
            solver_parameters,
            inputs={
                'data': ArrayKeys.RAW,
                'aff_label': ArrayKeys.GT_AFFINITIES,
            },
            outputs={
                'aff_pred': ArrayKeys.PRED_AFFINITIES
            },
            gradients={
                'aff_pred': ArrayKeys.LOSS_GRADIENT
            },
            use_gpu=gpu) +

        # save every 100th batch into an HDF5 file for manual inspection
        Snapshot({
            ArrayKeys.RAW: 'volumes/raw',
            ArrayKeys.GT_LABELS: 'volumes/labels/neuron_ids',
            ArrayKeys.GT_AFFINITIES: 'volumes/labels/gt_affinities',
            ArrayKeys.PRED_AFFINITIES: 'volumes/labels/pred_affinities',
            ArrayKeys.LOSS_GRADIENT: '/volumes/loss_gradient'
            },
            dataset_dtypes={
                ArrayKeys.GT_LABELS: np.uint64
            },
            every=100,
            output_dir='snapshots',
            output_filename='batch_{iteration}.hdf',
            additional_request=BatchRequest({ArrayKeys.LOSS_GRADIENT: request.arrays[ArrayKeys.GT_AFFINITIES]})) +

        # add useful profiling stats to identify bottlenecks
        PrintProfilingStats(every=10)
    )

    iterations = max_iteration - trained_until
    assert iterations >= 0

    print("Starting training...")
    with build(train_pipeline) as b:
        for i in range(iterations):
            b.request_batch(request)
    print("Training finished")

if __name__ == "__main__":
    iteration = int(sys.argv[1])
    gpu = int(sys.argv[2])
    train_until(iteration, gpu)
