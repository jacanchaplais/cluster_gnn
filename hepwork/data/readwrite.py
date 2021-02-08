import numpy as np
import pandas as pd
import vaex as vpd


def pcls_to_file(pcl_data, out_fname, data_id):
    """Takes particle data and saves it to HDF5. If writing to an
    existing file, data will be appended, and if with an existing key
    as well, data be added as additional rows to relevant table.

    Keyword arguments:
    pcl_data: (pd.DataFrame) containing mcpid and 4-momenta
    out_fname: (str) path in which to write data (must end '.h5')
    data_id: (str) key to data when retrieving from storage or appending
    """
    with pd.HDFStore(path=out_fname) as store:
        store.append(key=data_id, value=pcl_data, format='table',
                     append=True, data_columns=True)
        

def pcls_from_file(fname, data_id, where=None, start=None, stop=None,
                   columns=None, vaex=False):
    """Retrieve particle data stored in file, optionally based on where
    criteria.

    Keyword arguments:
    fname: (str) path to file storing the data (must be .h5)
    data_id: (str) id assigned to the data upon storage
    where: (list or None) Term (or convertible) objects, optional
    start: (int or None) event number to start selection
    stop: (int or None) event number to stop selection
    columns: (list or None) list of columns to return (None = all)
    vaex: (bool) reads a vaex formatted hdf5
    """
    if vaex:
        pass
    else:
        with pd.HDFStore(path=fname, mode='r') as store:
            return store.select(data_id, where, start, stop, columns,
                                auto_close=True)

def make_dataset(glob='/scratch/jlc1n20/data/*/Events/*/*.hdf5',
                 train_split=0.65, valid_split=0.15,
                 out='/home/jlc1n20/projects/particle_train/data/interim/'):
    data = vpd.open(glob)

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
    train_idx = int(np.floor(num_evts * train_split))
    valid_idx = train_idx + int(np.floor(num_evts * valid_split))
    test_idx = num_evts
    mask = {'train': data[col_name] < train_idx,
            'valid': (data[col_name] >= train_idx)
                      & (data[col_name] < valid_idx),
            'test': (data[col_name] >= valid_idx)
                     & (data[col_name] <= num_evts)
            }
    
    # write each dataset out to file
    for name, select in mask.items():
        data.export_hdf5(out + '/' + name + '.hdf5',
                         selection=select, virtual=False)

