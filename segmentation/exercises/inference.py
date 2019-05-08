from __future__ import print_function
from gunpowder import *
from gunpowder.tensorflow import *
import json
import numpy as np
import os
import sys
import logging
import daisy
import os


# os.environ["CUDA_VISIBLE_DEVICES"]="-1"

def predict(iteration, in_file, out_file, ):
    setup_dir = '/groups/funke/funkelab/sheridana/lsd_experiments/cremi/' \
                '02_train/setup04/'
    out_dataset = 'volumes/affs'

    # TODO: change to predict graph
    with open(os.path.join(setup_dir, 'train_net_config.json'), 'r') as f:
        config = json.load(f)

    # voxels
    voxel_size = Coordinate((40, 4, 4))
    input_shape = Coordinate(config['input_shape'])
    output_shape = Coordinate(config['output_shape'])
    print(output_shape)
    context = (input_shape - output_shape) // 2
    # print("Context is %s" % (context))

    chunkgrid = [2, 4, 4]
    # Initial ROI
    roi = Roi(
        offset=np.array((3800, 3800, 3400)) + np.array(
            (30, 400, 400)) * np.array(voxel_size),
        shape=output_shape * voxel_size * Coordinate(chunkgrid),
        # shape=output_shape * voxel_size
    )
    print("Original ROI %s" % roi)

    # nm

    context_nm = context * voxel_size
    read_roi = roi.copy()
    read_roi = read_roi.grow(context_nm, context_nm)
    print(read_roi.get_begin(), read_roi.get_end())

    # read_roi.snap_to_grid(input_shape * voxel_size)
    input_size = input_shape * voxel_size
    output_size = output_shape * voxel_size

    output_roi = read_roi.grow(-context_nm, -context_nm)
    print("Read ROI in nm is %s" % read_roi)
    print("Output ROI in nm is %s" % output_roi)

    print("Read ROI in voxel space is {}".format(read_roi / voxel_size))
    print("Output ROI in voxel space is {}".format(output_roi / voxel_size))

    # print("Cropped ROI in voxel space is {}".format(roi / voxel_size))

    raw = ArrayKey('RAW')
    affs = ArrayKey('AFFS')

    output_roi = daisy.Roi(
        output_roi.get_begin(),
        output_roi.get_shape()
    )

    ds = daisy.prepare_ds(
        out_file,
        out_dataset,
        output_roi,
        voxel_size,
        'float32',
        # write_size=output_size,
        write_roi=daisy.Roi((0, 0, 0), output_size),
        num_channels=3,
        # temporary fix until
        # https://github.com/zarr-developers/numcodecs/pull/87 gets approved
        # (we want gzip to be the default)
        compressor={'id': 'gzip', 'level': 5}
    )

    chunk_request = BatchRequest()
    chunk_request.add(raw, input_size)
    chunk_request.add(affs, output_size)

    pipeline = (
        N5Source(
            in_file,
            datasets = {
                raw: 'volumes/raw'
            },
        ) +
        Pad(raw, size=None) +
        Crop(raw, read_roi) +
        Normalize(raw) +
        IntensityScaleShift(raw, 2,-1) +
        Predict(
            os.path.join(setup_dir, 'train_net_checkpoint_%d'%iteration),
            inputs={
                config['raw']: raw
            },
            outputs={
                config['affs']: affs
            },
            # TODO: change to predict graph
            graph=os.path.join(setup_dir, 'train_net.meta')
        ) +
        IntensityScaleShift(raw, 0.5, 0.5) +
        ZarrWrite(
            dataset_names={
                affs: 'volumes/affs',
                raw: 'volumes/raw',
            },
            output_filename=out_file
        ) +
        PrintProfilingStats(every=10) +
        Scan(chunk_request)
    )

    print("Starting prediction...")
    with build(pipeline):
        pipeline.request_batch(BatchRequest())
    print("Prediction finished")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    logging.getLogger('gunpowder.nodes.hdf5like_write_base').setLevel(
        logging.DEBUG)

    in_file = '/groups/funke/funkelab/sheridana/lsd_experiments/' \
              'cremi/01_data/testing/sample_C_padded_20160501.' \
              'aligned.filled.cropped.62:153.n5'
    iteration = 500000
    out_file = 'affinities.zarr'

    predict(
        iteration,
        in_file,
        out_file)
