from __future__ import print_function
import sys; print(sys.path)
from gunpowder import *
from gunpowder.contrib import ZeroOutConstSections, PrepareMalis
import gunpowder.caffe
import gunpowder.tensorflow
import malis
import json
import glob
import math
import numpy as np
try:
    import tensorflow as tf
except:
    raise ImportWarning("tensorflow not available")

# the training HDF files
samples = [
    'trvol-250-1.hdf',
    # add more here
]

# after how many iterations to switch from Euclidean loss to MALIS
phase_switch = 500

def add_malis_loss(graph):
    with open('net_io_names.json', 'r') as f:
        net_io_names = json.load(f)
    affs = graph.get_tensor_by_name(net_io_names['affs'])
    gt_affs = graph.get_tensor_by_name(net_io_names['gt_affs'])
    gt_seg = tf.placeholder(tf.int64, shape=(92,)*3, name='gt_seg')

    loss = malis.malis_loss_op(affs, gt_affs, gt_seg, malis.mknhood3d())
    opt = tf.train.AdamOptimizer(
        learning_rate=0.5e-4,
        beta1=0.95,
        beta2=0.999,
        epsilon=1e-8,
        name='malis_optimizer')
    optimizer = opt.minimize(loss)

    return (loss, optimizer)


def train_until(max_iteration, gpu, backend):
    '''Resume training from the last stored network weights and train until ``max_iteration``.'''

    set_verbose(False)

    if backend == 'tf':
        if tf.train.latest_checkpoint('.'):
            trained_until = int(tf.train.latest_checkpoint('.').split('_')[-1])
        else:
            trained_until = 0
    elif backend == 'caffe':
        # get most recent training result
        solverstates = [int(f.split('.')[0].split('_')[-1]) for f in glob.glob('net_iter_*.solverstate')]
        if len(solverstates) > 0:
            trained_until = max(solverstates)
        else:
            trained_until = 0

    if trained_until == 0:
        print("Starting fresh training")
    else:
        print("Resuming training from iteration " + str(trained_until))

    if trained_until < phase_switch < max_iteration:
        # phase switch lies in-between, split training into to parts
        train_until(phase_switch, gpu, backend)
        trained_until = phase_switch

    if max_iteration <= phase_switch:
        phase = 'euclid'
    else:
        phase = 'malis'

    print("Traing until " + str(max_iteration) + " in phase " + phase)
    if backend == 'tf' or backend == 'tensorflow':
        with open('net_io_names.json', 'r') as f:
            net_io_names = json.load(f)
    elif backend == 'caffe':
        # setup training solver and network
        solver_parameters = gunpowder.caffe.SolverParameters()
        solver_parameters.train_net = 'net.prototxt'
        solver_parameters.base_lr = 0.5e-4
        solver_parameters.momentum = 0.95
        solver_parameters.momentum2 = 0.999
        solver_parameters.delta = 1e-8
        solver_parameters.weight_decay = 0.000005
        solver_parameters.lr_policy = 'inv'
        solver_parameters.gamma = 0.0001
        solver_parameters.power = 0.75
        solver_parameters.snapshot = 100
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
    request.add(ArrayKeys.GT_AFFINITIES_SCALE, output_shape)
    request.add(ArrayKeys.PRED_AFFINITIES, output_shape)

    if phase == 'malis' and backend == 'caffe':
        request.add(ArrayKeys.MALIS_COMP_LABEL, output_shape)

    # create a tuple of data sources, one for each HDF file
    data_sources = tuple(

        # provide arrays from the given HDF datasets
        Hdf5Source(
            sample,
            datasets={
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
                ArrayKeys.GT_MASK: Coordinate((100, 100, 100))*voxel_size
            }
        ) +

        # chose a random location inside the provided arrays
        RandomLocation() +

        # reject batches wich do contain less than 50% labelled data
        Reject(ArrayKeys.GT_MASK)

        for sample in samples
    )
    if backend == 'tf' or backend == 'tensorflow':
        train_inputs = {
            net_io_names['raw']: ArrayKeys.RAW,
            net_io_names['gt_affs']: ArrayKeys.GT_AFFINITIES,
            net_io_names['affs_loss_weights']: ArrayKeys.GT_AFFINITIES_SCALE
        }
        if phase == 'euclid':
            train_loss = net_io_names['loss']
            train_optimizer = net_io_names['optimizer']
        else:
            train_loss = None
            train_optimizer = add_malis_loss
            train_inputs['gt_seg:0'] = ArrayKeys.GT_LABELS
        train_node = gunpowder.tensorflow.Train(
            'unet',
            optimizer=train_optimizer,
            loss=train_loss,
            inputs=train_inputs,
            outputs={
                net_io_names['affs']: ArrayKeys.PRED_AFFINITIES
            },
            gradients={net_io_names['affs']: ArrayKeys.LOSS_GRADIENT},
            summary=net_io_names['summary'],
            save_every=100
        )
    elif backend == 'caffe':
        train_inputs = {'data': ArrayKeys.RAW,
                        'aff_label': ArrayKeys.GT_AFFINITIES}

        if phase == 'euclid':

            train_inputs['scale']=ArrayKeys.GT_AFFINITIES_SCALE

        elif phase == 'malis':

            train_inputs['comp_label']=ArrayKeys.MALIS_COMP_LABEL
            train_inputs['nhood']='affinity_neighborhood'

        train_node = gunpowder.caffe.Train(
            solver_parameters,
            inputs=train_inputs,
            outputs={'aff_pred': ArrayKeys.PRED_AFFINITIES},
            gradients={'aff_pred': ArrayKeys.LOSS_GRADIENT},
            use_gpu=gpu)


    # attach data sources to training pipeline
    prepare_pipeline = (

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
                        gt_affinities_mask=ArrayKeys.GT_AFFINITIES_MASK) )
    if backend == 'caffe' and phase == 'malis':
        prepare_pipeline = (prepare_pipeline +
            PrepareMalis(
                labels_array_key=ArrayKeys.GT_LABELS,
                malis_comp_array_key=ArrayKeys.MALIS_COMP_LABEL
            ))
    train_pipeline = (prepare_pipeline+
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
        train_node +

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
            additional_request=BatchRequest({ArrayKeys.LOSS_GRADIENT: request[ArrayKeys.GT_AFFINITIES]})) +

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
    backend = str(sys.argv[3])
    train_until(iteration, gpu, backend)
