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

import click

from hepwork.data import io


@click.command()

@click.argument('in_path', type=click.Path(exists=True, resolve_path=True))
@click.argument('num_evts', type=click.INT)
@click.argument('mcpids', type=click.IntRange(min=1, max=9999999), nargs=-1)

@click.option('-s', '--stride', default=1000, show_default=True,
              type=click.IntRange(min=10, max=5000, clamp=True))
@click.option('-n', '--num-procs', type=click.IntRange(min=1), default=1,
              show_default=True)
@click.option('-o', '--out-path', type=click.Path())
@click.option('--overwrite', is_flag=True)

def convert(in_path, num_evts, mcpids, stride, num_procs, out_path, overwrite):
    """Converts raw HepMC data into HDF5 dataframe of clustered jets."""

    if (out_path is None):
        base_fname = in_path.split('.')[0]
        out_path = base_fname + '.h5'

    data = None
    data = io.jets_from_raw(in_path, num_evts, mcpids, stride, num_procs)

    if (overwrite and (data is not None) and os.path.exists(out_path)):
        os.remove(out_path)
    
    io.pcls_to_file(pcl_data=data, out_fname=out_path, data_id='jet_data')


if __name__ == '__main__':
    convert()