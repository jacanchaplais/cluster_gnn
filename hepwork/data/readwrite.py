import os
import warnings

import numpy as np
import pandas as pd
import vaex as vpd


def pcls_to_file(pcl_data, out_fname):
    """Takes Vaex particle data and saves it to HDF5. 
    """
    if (os.path.exists(out_fname)):
        warnings.warn("File {} already exists, overwriting.".format(out_fname))
        os.remove(out_fname)
    pcl_data.export_hdf5(out_fname)
        

def pcls_from_file(fname):
    """Retrieve particle data stored in file as Vaex dataframe.

    Note: if filepath can be globbed to match multiple files, data from
    every matching file will be returned in a single dataframe.

    Keyword arguments:
    fname: (str) path to file storing the data (must be .h5)
    """
    return vpd.open(fname)

def make_dataset(glob='/scratch/jlc1n20/data/*/Events/*/*.hdf5',
                 splits=[0.65, 0.15, 0.2],
                 out='/home/jlc1n20/projects/particle_train/data/interim/'):
    data = pcls_from_file(glob)

    # shuffle signal and bg events so they are in random order
    col_name = 'event'
    evts, idxs = np.unique(data[col_name].values, return_inverse=True)
    num_evts = len(evts)
    evts = np.arange(0, num_evts, dtype=np.uint32)
    rng = np.random.default_rng()
    rng.shuffle(evts)
    data[col_name] = evts[idxs]
    data = data.sort(col_name)

    # expressions to split data into train, valid and test sets
    mask = {}
    mask['train'], mask['valid'], mask['test'] = data.split(frac=splits)
    
    # write each dataset out to file
    for name, select in mask.items():
        data.export_hdf5(out + '/' + name + '.hdf5',
                         selection=select, virtual=False)

