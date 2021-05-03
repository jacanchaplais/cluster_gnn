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

