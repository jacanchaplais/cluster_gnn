import os

import click
import h5py
import vaex as vpd
import numpy as np

@click.group()
@click.argument('in_path', type=click.Path(exists=True, resolve_path=True))
@click.argument('out_path', type=click.Path(exists=False, file_okay=True,
                                            resolve_path=True, writable=True,
                                            readable=False))
@click.pass_context
def format(ctx, in_path, out_path):
    dpath = lambda dname: 'table/columns/' + dname + '/data'
    if os.path.exists(out_path):
        os.remove(out_path)
    with h5py.File(in_path, 'r') as f_in:
        evts, evt_cts = np.unique(f_in[dpath('event')], return_counts=True)

        # create virtual dataset stacking the 4-momenta for easy access
        p_src = []
        for name in ['energy', 'px', 'py', 'pz']:
            p_src.append(h5py.VirtualSource(f_in[dpath(name)]))
        p_lay = h5py.VirtualLayout(shape=(4,)+p_src[0].shape, dtype='<f8')
        for i in range(len(p_src)):
            p_lay[i, :] = p_src[i][:]

        temp_path = '/home/jlc1n20/projects/particle_train/data/interim/sandbox/temp.hdf5'

        f_temp = h5py.File(temp_path, 'w', libver='latest')
        f_temp.create_virtual_dataset('vpmu', p_lay)

        # export context to children
        ctx.obj['evt_cts'] = evt_cts
        ctx.obj['in_path'] = in_path
        ctx.obj['out_path'] = out_path
        ctx.obj['pmu'] = f_temp['vpmu']

    @ctx.call_on_close
    def cleanup():
        f_temp.close()
        os.remove(temp_path)

@format.command()
@click.pass_context
def lgn(ctx):
    in_path = ctx.obj['in_path']
    out_path = ctx.obj['out_path']
    evt_cts = ctx.obj['evt_cts'] 
    num_evts = len(evt_cts)
    chunks = np.cumsum(evt_cts)
    chunks = np.insert(chunks, 0, 0)
    num_pcls = np.sum(evt_cts)

    pmu = ctx.obj['pmu']
    dsets = {
        # dataset properties relating to the whole jet
        # 'Nobj': {
        #     'shape': (num_evts,),
        #     'compression': None,
        #     'shuffle': False,
        #     'dtype': '<i2'},
        # 'is_signal': {
        #     'shape': (num_evts,),
        #     'compression': None,
        #     'shuffle': False,
        #     'dtype': '<i2'},
        # 'jet_pt': {
        #     'data': jet_pt,
        #     'shape': (num_evts,),
        #     'compression': None,
        #     'shuffle': False,
        #     'dtype': '<f8',},
        # 'truth_Pmu': {
        #     'shape': (num_evts, 4),
        #     'compression': None,
        #     'shuffle': False,
        #     'slice': (),
        #     'dtype': '<f8',},

        # # dataset properties relating to jet constituents
        # 'label': {
        #     'data': np.ones(200, dtype='<i2'),
        #     'shape': (num_evts, 200),
        #     'compression': 'lzf',
        #     'shuffle': True,
        #     'slice': (),
        #     'chunk_size': (1, 200),
        #     'dtype': '<i2',},
        # 'mass': {
        #     'shape': (num_evts, 200),
        #     'compression': 'lzf',
        #     'shuffle': True,
        #     'chunk_size': (1, 200),
        #     'dtype': '<f8',},
        'Pmu': {
            'data': pmu,
            'shape': (num_evts, 200, 4),
            'compression': 'lzf',
            'shuffle': True,
            'slice': (slice(None),),
            'chunk_size': (1, 200, 4),
            'dtype': '<f8',},
    }

    with h5py.File(out_path, 'w', libver='latest') as f_out:
        for name, info in dsets.items():
            # for key, val in info.items():
            #     click.echo(key + ': {}'.format(val))
            # return
            f_out.create_dataset(
                name, dtype=info['dtype'], shape=info['shape'],
                chunks=info['chunk_size'], compression=info['compression'],
                shuffle=info['shuffle'])
            dset = f_out[name]

            data = info['data']

            for evt in range(1, num_evts + 1):
                start = chunks[evt - 1]
                stop = chunks[evt]
                num_in_jet = stop - start
                # dset_idx = (evt - 1, slice(None, num_in_jet),) \
                #            + info['slice'] 
                # dsrc_idx = info['slice'] + (slice(start, stop),)
                # click.echo('dset idx: {}'.format(dset_idx))
                # click.echo('dsrc idx: {}'.format(dsrc_idx))
                # return
                dset[evt - 1, :num_in_jet, :] = data[:, start:stop].T
            # elif name is 'label':
            #     for evt in range(1, num_evts + 1):
            #         start = chunks[evt - 1]
            #         stop = chunks[evt]
            #         num_in_jet = stop - start
            

@format.command()
@click.pass_context
def efn():
    pass

if __name__ == '__main__':
    format(obj={})
