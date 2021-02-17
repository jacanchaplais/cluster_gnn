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
@click.argument('tag_mcpid', type=click.INT)
@click.option('--overwrite/--no-overwrite', default=True)
@click.pass_context
def format(ctx, in_path, out_dir, tag_mcpid, overwrite):
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
        # counting the number of particles per event
        evts, evt_cts = np.unique(f_in[dkey('event')], return_counts=True)
        num_evts = len(evt_cts)
        parent = f_in[dkey('parent')][...]

        # check that the number of parents is equal to number of events
        if (num_evts != np.sum(parent, dtype=int)):
            raise ValueError(
                "Mismatch between number of parents and number of jets")

        # STORE DATA IN TEMP FILE:
        # temp file creation
        tmp_path = out_dir + '/.tmp'
        f_temp = h5py.File(tmp_path, 'w', libver='latest')
        # pmu virtual dset source defn and stacking in layout
        pmu = {'energy': None, 'px': None, 'py': None, 'pz': None}
        pmu_src = []
        for name in pmu:
            pmu[name] = f_in[dkey(name)] # saving in dict for other calcs
            pmu_src.append(h5py.VirtualSource(pmu[name]))
        pmu_lyt = h5py.VirtualLayout(shape=(4,)+pmu_src[0].shape, dtype='<f8')
        for i in range(len(pmu_src)):
            pmu_lyt[i, :] = pmu_src[i][:]
        # writing data out
        f_temp.create_virtual_dataset('vpmu', pmu_lyt) # stacked pmu
        f_temp.create_dataset( # jet pTs (using parent pcl pT)
            name='pt', dtype=pmu['px'].dtype, shape=(num_evts,),
            data=np.sqrt(pmu['px'][parent] ** 2 + pmu['py'][parent] ** 2))

        mass_sq = (pmu['energy'][...] ** 2
                   - pmu['px'][...] ** 2
                   - pmu['py'][...] ** 2
                   - pmu['pz'][...] ** 2)
        mass_sq = np.where(mass_sq > 0.0, mass_sq, 0.0)
        f_temp.create_dataset( # mass of constituents
            name='mass', dtype=pmu['px'].dtype, shape=pmu['px'].shape,
            data=np.sqrt(mass_sq))
        del mass_sq

        # STORE DATA IN CONTEXT OBJ:
        mcpid = f_in[dkey('mcpid')]
        is_signal = mcpid[parent] == tag_mcpid
        # export context to subcommands
        ctx.obj['evt_cts'] = evt_cts
        ctx.obj['in_path'] = in_path
        ctx.obj['out_path'] = out_path
        ctx.obj['pmu'] = f_temp['vpmu']
        ctx.obj['pt'] = f_temp['pt']
        ctx.obj['mass'] = f_temp['mass']
        ctx.obj['parent'] = parent
        ctx.obj['is_signal'] = is_signal

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
    pt = ctx.obj['pt']
    mass = ctx.obj['mass']
    parent_mask = ctx.obj['parent']
    is_signal = ctx.obj['is_signal']
    evt_cts = ctx.obj['evt_cts'] 

    # counting and finding locations of particle / event splits
    num_evts = len(evt_cts)
    num_pcls = np.sum(evt_cts)
    chops = np.cumsum(evt_cts)
    chops = np.insert(chops, 0, 0)

    # dataset properties in easy to change dict interface
    dsets = {
        # dataset properties relating to the whole jet
        'Nobj': {
            'data': evt_cts - 1, # num pcls -1 for parent
            'shape': (num_evts,),
            'compression': None,
            'chunk_size': None,
            'shuffle': False,
            'slice': (),
            'dtype': '<i2'},
        'is_signal': {
            'data': is_signal,
            'shape': (num_evts,),
            'compression': None,
            'chunk_size': None,
            'shuffle': False,
            'slice': (),
            'dtype': '<i2'},
        'jet_pt': {
            'data': pt,
            'shape': (num_evts,),
            'compression': None,
            'chunk_size': None,
            'shuffle': False,
            'slice': (),
            'dtype': '<f8',},
        'truth_Pmu': {
            'data': pmu,
            'shape': (num_evts, 4),
            'compression': None,
            'chunk_size': None,
            'shuffle': False,
            'slice': (),
            'dtype': '<f8',},

        # # dataset properties relating to jet constituents
        'label': {
            'data': None,
            'shape': (num_evts, 200),
            'compression': 'lzf',
            'shuffle': True,
            'slice': (),
            'chunk_size': (1, 200),
            'dtype': '<i2',},
        'mass': {
            'data': mass,
            'shape': (num_evts, 200),
            'compression': 'lzf',
            'shuffle': True,
            'slice': (),
            'chunk_size': (1, 200),
            'dtype': '<f8',},
        'Pmu': {
            'data': pmu,
            'shape': (num_evts, 200, 4),
            'compression': 'lzf',
            'shuffle': True,
            'slice': (slice(None),),
            'chunk_size': (1, 200, 4),
            'dtype': '<f8',},
    }
    
    def chop_info(evt, chops):
        """Returns info for iterating contiguous data chopped into evts.

        Keyword args:
        evt: (int) event num to get the chop info for
        chops: (1d array) starting at 0, idxs of evt start / stop points

        Returns:
        (tuple): start, stop, num final pcls, idx for writing to dest
        """
        start = chops[evt - 1]
        stop = chops[evt]
        num_in_jet = stop - start - 1
        dset_idx = (evt - 1, slice(None, num_in_jet),)
        return (start, stop, num_in_jet, dset_idx,)

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
            # TODO: make this DRY without if within for
            evt_range = range(1, num_evts + 1)

            if name.lower() == 'pmu':
                for evt in evt_range:
                    start, stop, num_in_jet, dset_idx = chop_info(evt, chops)
                    dset_idx = dset_idx + slc
                    # select only children
                    mask = parent_mask[start:stop] == False
                    dsrc_idx = slc + (slice(start, stop),)
                    # pmu shape has to be transposed
                    dset[dset_idx] = data[dsrc_idx][:, mask].T
            elif name.lower() == 'truth_pmu':
                mask = parent_mask
                dset[...] = data[...][:, mask].T
            elif name.lower() == 'mass':
                for evt in evt_range:
                    start, stop, num_in_jet, dset_idx = chop_info(evt, chops)
                    # select only children
                    mask = parent_mask[start:stop] == False
                    dsrc_idx = slice(start, stop)
                    dset[dset_idx] = data[dsrc_idx][mask]
            elif name.lower() == 'label':
                for evt in evt_range:
                    start, stop, num_in_jet, dset_idx = chop_info(evt, chops)
                    dset[dset_idx] = 1
            else:
                dset[...] = data[...]


@format.command()
@click.pass_context
def efn():
    pass

if __name__ == '__main__':
    format(obj={})
