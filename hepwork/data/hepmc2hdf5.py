import sys
import os

import numpy as np
import h5py

from jet_tools.src import FormJets, Components, TrueTag, ReadHepmc

#------------------------------- READ IN HEPMC -------------------------------#
NUM_EVENTS = int(1e+2)
NUM_EVENTS_PER_CHUNK = 100
NUM_PCLS = 200
NUM_TRUTH_PCLS = 10
SIGNAL_MCPID = 6
JET_NAME = "TopJets"
dR = 1.0

in_fname = 'data/raw/5E4/tag_1_pythia8_events.hepmc'
out_fname = 'data/interim/1E2_test_2.h5'

#------------------------- DEFINE HDF5 DATA STRUCTURE ------------------------#
data = {
    'Nobj': # number of jet constituents
    np.zeros(NUM_EVENTS_PER_CHUNK, dtype=np.dtype('i2')),
    
    'Pmu': # 4-momenta of jet constituents
    np.zeros((NUM_EVENTS_PER_CHUNK, NUM_PCLS, 4),
              dtype=np.dtype('f8')),

    'truth_Nobj': # number of truth-level particles
    np.zeros(NUM_EVENTS_PER_CHUNK, dtype=np.dtype('i2')),

    'truth_Pdg': # PDG codes to ID truth particles
    np.zeros((NUM_EVENTS_PER_CHUNK, NUM_TRUTH_PCLS),
              dtype=np.dtype('i4')),

    'truth_Pmu': # truth-level particle 4-momenta
    np.zeros((NUM_EVENTS_PER_CHUNK, NUM_TRUTH_PCLS, 4),
              dtype=np.dtype('f8')),

    'is_signal': # signal flag (0 = background, 1 = signal)
    np.zeros(NUM_EVENTS_PER_CHUNK, dtype=np.dtype('i1'))
}

#-------------------------- CREATE EMPTY HDF5 FILE ---------------------------#
dsets = {}
with h5py.File(out_fname, 'w') as f:
    for key, val in data.items():
        shape = list(val.shape)
        shape[0] = NUM_EVENTS
        shape = tuple(shape)
        dsets[key] = f.create_dataset(
                key, shape, val.dtype, compression='gzip')

#------------------------------ WRITING TO HDF5 ------------------------------#
# Some indexing preparation
num_chunks = int(np.ceil(NUM_EVENTS / NUM_EVENTS_PER_CHUNK))
start_idxs = np.zeros(num_chunks, dtype=np.dtype('i8'))

for i in range(1, start_idxs.shape[0]):
    start_idxs[i] = start_idxs[i-1] + NUM_EVENTS_PER_CHUNK

stop_idxs = start_idxs + NUM_EVENTS_PER_CHUNK
stop_idxs[-1] = NUM_EVENTS
ranges = stop_idxs - start_idxs

print('Writing to HDF5 file in ' + str(num_chunks) + ' chunks.')

#------------------------------- READ IN HEPMC -------------------------------#
for chunk_idx in range(num_chunks):
    hep_data = ReadHepmc.Hepmc(
            in_fname,
            start=start_idxs[chunk_idx],
            stop=stop_idxs[chunk_idx])

    # calculate higher level (HL) data
    Components.add_all(hep_data, inc_mass=True)

    # filter using the HL data
    filt_fin_pcl = FormJets.filter_ends # pcls at end of shower
    filt_observe = FormJets.filter_pt_eta # observable pt & eta range
    FormJets.create_jetInputs( # apply the filters to edit data inplace
            hep_data,
            [filt_fin_pcl, filt_observe],
            batch_length=np.inf)

    # jet cluster events with anti-kt
    FormJets.cluster_multiapply(
            hep_data,
            cluster_algorithm=FormJets.Traditional,
            dict_jet_params={'DeltaR': dR, 'ExpofPTMultiplier': -1},
            jet_name=JET_NAME,
            batch_length=np.inf)

   # clear the buffer (for safety)
    for key in data.keys(): data[key][:] = 0

    for j in range(hep_data.n_events):
        hep_data.selected_index = j

        tag_idx = TrueTag.tag_particle_indices(
                hep_data,
                tag_pids=[SIGNAL_MCPID],
                include_antiparticles=False)

        lead_jet_idx = TrueTag.allocate(hep_data, JET_NAME, tag_idx, dR**2)

        jet_pcls_idx = getattr(hep_data, JET_NAME + "_Child1")[lead_jet_idx][0]
        constit_mask = jet_pcls_idx == -1
        num_jet_constits = np.sum(constit_mask)

        constit_pmu = {}
        for key in ["Energy", "Px", "Py", "Pz"]:
            temp = getattr(hep_data, JET_NAME + "_" + key)
            constit_pmu[key] = temp[lead_jet_idx][0][constit_mask]

        del temp
        
        data['Nobj'][j] = num_jet_constits

        for k in range(num_jet_constits):
            data['Pmu'][j,k,0] = constit_pmu["Energy"][k]
            data['Pmu'][j,k,1] = constit_pmu["Px"][k]
            data['Pmu'][j,k,2] = constit_pmu["Py"][k]
            data['Pmu'][j,k,3] = constit_pmu["Pz"][k]

#----------------------- ADDITIONAL "TRUTH" PARTICLES ------------------------#
# optionally store additional information about "truth" pcls, eg.
# mother particle of jet, intermediate states, etc.
        
        truth_idx = np.array(tag_idx, dtype=np.int)
        num_truth = len(truth_idx)
        data['truth_Nobj'][j] = num_truth

        # sort the particles in terms of their MCPID / pdg:
        truth_seq = np.argsort(np.abs(hep_data.MCPID[truth_idx]))
        truth_idx_sort = truth_idx[truth_seq]
        
        k = 0
        while(k < num_truth and k < NUM_TRUTH_PCLS):
            idx = int(truth_idx_sort[k])
            data['truth_Pmu'][j,k,0] = hep_data.Energy[idx]
            data['truth_Pmu'][j,k,1] = hep_data.Px[idx]
            data['truth_Pmu'][j,k,2] = hep_data.Py[idx]
            data['truth_Pmu'][j,k,3] = hep_data.Pz[idx]
            data['truth_Pdg'][j,k] = hep_data.MCPID[idx]
            
            k = k + 1

        data['is_signal'][j] = hep_data.MCPID[tag_idx] == SIGNAL_MCPID

    print('\tWriting chunk ' + str(chunk_idx) + '.')

    with h5py.File(out_fname, 'a') as f:
        for key in dsets.keys():
            dset = f[key]
            start, stop = start_idxs[chunk_idx], stop_idxs[chunk_idx]
            dset[start:stop] = data[key][:ranges[chunk_idx]]

