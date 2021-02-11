import click
import h5py
import vaex as vpd
import numpy as np

@click.group()
@click.argument('in_path', type=click.Path(exists=True, resolve_path=True))
@click.pass_context
def format(ctx, in_path):
    fd = h5py.File(in_path, 'r')
    dpath = lambda dname: 'table/columns/' + dname + '/data'
    evts, evt_cts = np.unique(fd[dpath('event')], return_counts=True)

    ctx.obj['pmu'] = {}
    for name in ['energy', 'px', 'py', 'pz']:
        ctx.obj['pmu'][name] = fd[dpath(name)]

    ctx.obj['evt_cts'] = evt_cts
    ctx.obj['par_mask'] = fd[dpath('parent')][...]

@format.command()
@click.pass_context
def lgn(ctx):
    evt_cts = ctx.obj['evt_cts'] 
    num_evts = len(evt_cts)
    chunks = np.cumsum(evt_cts)
    chunks = np.insert(chunks, 0, 0)
    num_pcls = np.sum(evt_cts)
    pmu = list(ctx.obj['pmu'].values())

    dsets = {
        # 'Nobj': {
        #     'shape': (num_evts,),
        #     'dtype': '<i2'
        #     },
        # 'label': {
        #     'shape': (num_evts, 200),
        #     'dtype': '<i2'
        #     },
        # 'is_signal': {
        #     'shape': (num_evts,),
        #     'dtype': '<i2'
        #     # 'data': par_df['(mcpid == 6)']
        #     },
        # 'jet_pt': {
        #     'shape': (num_evts,),
        #     'dtype': '<f8',
        #     'data': jet_pt
        #     },
        # 'mass': {
        #     'shape': (num_evts, 200),
        #     'dtype': '<f8'
        #     }
        'Pmu': {
            'shape': (num_evts, 200, 4),
            'dtype': '<f8',
            'data': pmu
            },
        # 'truth_Pmu': {
        #     'shape': (num_evts, 4),
        #     'dtype': '<f8'
        #     }
            }
    
    with h5py.File('temp.hdf5', 'w', libver='latest') as f:
        for name, info in dsets.items():
            f.create_dataset(name, fillvalue=0.0, dtype=info['dtype'],
                             shape=info['shape'])
            dset = f[name]
            for evt in range(1, num_evts + 1):
                start = chunks[evt - 1]
                stop = chunks[evt]
                num_in_jet = stop - start
                for i in range(len(info['data'])):
                    dset[evt - 1, :num_in_jet, i] = pmu[i][start:stop]

@format.command()
@click.pass_context
def efn():
    pass

if __name__ == '__main__':
    format(obj={})
