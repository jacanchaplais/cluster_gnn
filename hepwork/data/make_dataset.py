###############################################################################
#  Raw to HDF5 converter                                                      #
#                                                                             #
#  Automatically cluster raw MadGraph output data into jets, and convert      #
#  into HDF5 files with one jet per event.                                    #
#                                                                             #
#  Usage:                                                                     # 
#  Run `python convert.py --help` to see description, args and flags.         #
#                                                                             #
#  Author:                                                                    #
#  Written by Jacan Chaplais, 27/01/2021                                      #
###############################################################################

import os
import glob

import click
import numpy as np
import vaex as vpd

from hepwork.data import readwrite as rw

@click.group()
def make_dataset():
    pass

@make_dataset.command()
@click.argument('in_path', type=click.Path(exists=True, resolve_path=True))
@click.argument('num_evts', type=click.INT)
@click.argument('mcpids', type=click.IntRange(min=1, max=9999999), nargs=-1)
@click.option('-s', '--stride', default=1000, show_default=True,
              type=click.IntRange(min=10, max=5000, clamp=True))
@click.option('--offset', default=0, show_default=True,
              type=click.IntRange(min=0))
@click.option('-n', '--num-procs', type=click.IntRange(min=1), default=1,
              show_default=True)
@click.option('--out-path', type=click.Path())
@click.option('--overwrite', is_flag=True)
def extract(in_path, num_evts, mcpids, stride, offset, num_procs, out_path,
            overwrite):
    """Converts raw HepMC data into HDF5 dataframe of clustered jets."""
    # scoped import of slow module to prevent slowing other commands
    from hepwork.data import process 

    if (out_path is None):
        base_fname = in_path.split('.')[0]
        out_path = base_fname + '.hdf5'

    data = None
    data = process.jets_from_raw(in_path, num_evts, mcpids, stride, num_procs)

    if (overwrite and (data is not None) and os.path.exists(out_path)):
        os.remove(out_path)
    
    data['event'] = data['event'] + offset
    rw.pcls_to_file(data, out_path)


@make_dataset.command()
@click.argument('glob', type=click.Path())
@click.argument('out_dir', type=click.Path(exists=True, file_okay=False,
                readable=False, writable=True))
@click.option('-s', '--splits', type=click.FLOAT, multiple=True,
              default=[0.65, 0.15, 0.2], show_default=True,
              help="Ratio of events used in train : valid : test datasets")
def merge(glob, out_dir, splits):
    """Combines jets data from each of the simulation runs into train,
    validate, and test datasets, ready for processing by a neural net.
    GLOB provides the path pattern matching all HDF5 files to combine.
    OUT_DIR is the Location into which the datasets are written.
    """
    data = rw.pcls_from_file(glob)

    # shuffle signal and bg events so they are in random order
    col_name = 'event'
    evts, idxs = np.unique(data[col_name].values, return_inverse=True)
    num_evts = len(evts)
    evts = np.arange(0, num_evts, dtype=np.uint32)
    rng = np.random.default_rng()
    rng.shuffle(evts)
    data[col_name] = evts[idxs]
    data = data.sort(col_name)

    sum_spl = float(sum(splits))
    tr_spl = splits[0] / sum_spl
    val_spl = splits[1] / sum_spl

    # expressions to split data into train, valid and test sets
    train_idx = int(np.floor(num_evts * tr_spl))
    valid_idx = train_idx + int(np.floor(num_evts * val_spl))
    test_idx = num_evts
    mask = {'train': data[col_name] < train_idx,
            'valid': (data[col_name] >= train_idx)
                      & (data[col_name] < valid_idx),
            'test': (data[col_name] >= valid_idx)
                     & (data[col_name] <= num_evts)
            }
    
    # write each dataset out to file
    for name, select in mask.items():
        data.export_hdf5(out_dir + '/' + name + '.hdf5', selection=select,
                         virtual=False, progress=True)

@make_dataset.command()
@click.argument('pattern', type=click.str)
@click.argument('offset', type=click.Int)
def reindex(pattern, offset):
    """Adds offset to event index for fragment hdf5 files matching glob
    pattern. Useful if you forgot to set offset when first extracting.
    """
    for fpath in glob.iglob(pattern):
        # create path for temporary converted file
        fname, fext = os.path.splitext(fpath)
        new_fpath = fname + '_c' + fext

        # get data and apply the offset
        vdf = vpd.open(fpath)
        vdf['event'] = vdf['event'] + offset
        vdf.export_hdf5(new_fpath)
        vdf.close()

        # replace the old file with the reindexed one
        os.replace(new_fpath, fpath)


if __name__ == '__main__':
    make_dataset()
