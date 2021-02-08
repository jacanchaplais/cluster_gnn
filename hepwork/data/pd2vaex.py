import os

import click
import vaex as vpd

from hepwork.data import readwrite as rw


@click.command()

@click.argument('in_path', type=click.Path(exists=True, resolve_path=True))
@click.option('--out-path', type=click.Path())
@click.option('--offset', default=0, show_default=True,
              type=click.IntRange(min=0))

def pd2vaex(in_path, out_path, offset):
    """Converts pandas HDF5 into Vaex HDF5"""

    if (out_path is None):
        base_fname = in_path.split('.')[0]
        out_path = base_fname + '_vaex.h5'

    # data = None
    df = rw.pcls_from_file(in_path, 'jet_data')
    vdf = vpd.from_pandas(df, copy_index=True, index_name='event')
    vdf['event'] = vdf['event'] + offset
    vdf.export_hdf5(out_path)


if __name__ == '__main__':
    pd2vaex()
