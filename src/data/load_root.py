import pprint

import uproot as rt


data_dir = '/scratch/jlc1n20/data/sm_tpair3/Events/05_run'
root_file = 'unweighted_events.root'
root_path = data_dir + '/' + root_file

pp = pprint.PrettyPrinter(indent=4)

with rt.open(root_path) as f:
    tree = f['LHEF']
    pp.pprint(tree.allkeys())
    pt = tree['Particle.PT'].array() # extract data from key as an array

pp.pprint(pt)

