import os

import click
import h5py
import vaex as vpd
import numpy as np

@click.group()
@click.argument('in_path', type=click.Path(exists=True, resolve_path=True))
@click.argument('out_dir', type=click.Path(exists=False, dir_okay=True,
                                            resolve_path=True, writable=True,
                                            readable=False))
@click.option('--overwrite/--no-overwrite', default=True)
@click.pass_context
def format(ctx, in_path, out_dir, overwrite):
    # preparing the path of the new file:
    f_name = os.path.basename(in_path)
    f_name, f_ext = os.path.splitext(f_name)
    out_path = out_dir + '/' + f_name + '_c' + f_ext
    if os.path.exists(out_path):
        if overwrite:
            os.remove(out_path)
        else:
            raise FileExistsError(
                "A converted file of the same name exists in this "
                + "location. Please move / rename it, or run again "
                + "without the --no-overwrite flag.")

    # extracting data and passing it to context for subcommands:
    dkey = lambda dname: 'table/columns/' + dname + '/data'
    with h5py.File(in_path, 'r') as f_in:
        # create virtual dataset stacking the 4-momenta for easy access
        pmu_src = []
        for name in ['energy', 'px', 'py', 'pz']:
            pmu_src.append(h5py.VirtualSource(f_in[dkey(name)]))
        pmu_lyt = h5py.VirtualLayout(shape=(4,)+pmu_src[0].shape, dtype='<f8')
        for i in range(len(pmu_src)):
            pmu_lyt[i, :] = pmu_src[i][:]

        # writing out virtual pmu to temporary file
        tmp_path = out_dir + '/.tmp'
        f_temp = h5py.File(tmp_path, 'w', libver='latest')
        f_temp.create_virtual_dataset('vpmu', pmu_lyt)

        # counting the number of particles per event
        evts, evt_cts = np.unique(f_in[dkey('event')], return_counts=True)

        # export context to subcommands
        ctx.obj['evt_cts'] = evt_cts
        ctx.obj['in_path'] = in_path
        ctx.obj['out_path'] = out_path
        ctx.obj['pmu'] = f_temp['vpmu']

    @ctx.call_on_close
    def cleanup():
        # deleting temporary file holding virtual pmu dset
        f_temp.close()
        os.remove(tmp_path)

@format.command()
@click.pass_context
def lgn(ctx):
    # retrieving data from context
    in_path = ctx.obj['in_path']
    out_path = ctx.obj['out_path']
    pmu = ctx.obj['pmu']
    evt_cts = ctx.obj['evt_cts'] 
    # counting and finding locations of particle / event splits
    num_evts = len(evt_cts)
    num_pcls = np.sum(evt_cts)
    chunks = np.cumsum(evt_cts)
    chunks = np.insert(chunks, 0, 0)

    # dataset properties in easy to change dict interface
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
        'label': {
            'data': np.ones(200, dtype='<i2'),
            'shape': (num_evts, 200),
            'compression': 'lzf',
            'shuffle': True,
            'slice': (),
            'chunk_size': (1, 200),
            'dtype': '<i2',},
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

    # writing the datasets out to file
    with h5py.File(out_path, 'w', libver='latest') as f_out:
        for name, info in dsets.items():
            f_out.create_dataset(
                name, dtype=info['dtype'], shape=info['shape'],
                chunks=info['chunk_size'], compression=info['compression'],
                shuffle=info['shuffle'])

            # data from keys before intensive loop (big overhead if not)
            dset = f_out[name]
            data = info['data']
            slc = info['slice']

            # writing out constit properties to each event manually
            # TODO: check if faster by using np.split and creating zero
            #       padded array, then using vectorised assignment
            # TODO: make this DRY without if within for
            evt_range = range(1, num_evts + 1)
            if name.lower() == 'pmu':
                for evt in evt_range:
                    # TODO: use np.ma mask arrays to identify parent pcl
                    start = chunks[evt - 1]
                    stop = chunks[evt]
                    num_in_jet = stop - start
                    # defn slices outside use to enable diffs between dsets:
                    dset_idx = (evt - 1, slice(None, num_in_jet),) + slc
                    dsrc_idx = slc + (slice(start, stop),)
                    # pmu shape has to be transposed
                    dset[dset_idx] = data[dsrc_idx].T
            elif name.lower() == 'label':
                for evt in evt_range:
                    start = chunks[evt - 1]
                    stop = chunks[evt]
                    num_in_jet = stop - start
                    # defn slices outside use to enable diffs between dsets:
                    dset_idx = (evt - 1, slice(None, num_in_jet),)
                    dset[dset_idx] = 1

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
