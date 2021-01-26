import numpy as np
import pandas as pd
import h5py

from jet_tools.src import ReadHepmc, Components, FormJets, TrueTag


def jets_from_raw(in_fname, num_evts, stride=1000, tag_mcpid=[6]):
    """Returns a pandas dataframe of jet constituent calorimeter info.
    
    Keyword arguments:
    in_fname: (str) path to file containing data
    interv: (list) events to read from file, ie. [start, end]
    tag_mcpid: (array like) list of ancestor particle ids forming jets
    """

    num_chunks = int(np.ceil(num_evts / stride))
    starts = np.arange(0, num_evts, stride, dtype=int)

    ranges = np.array([0, stride - 1]).reshape(1, 2) + starts[:, np.newaxis]
    ranges[-1, 1] = num_evts
    
    for interv in ranges:
        hep_data = HepData(in_fname, interv)
        cur_jet = hep_data.get_jet_constits(tag_mcpid)
        if (interv[0] == 0):
            jets = cur_jet
        else:
            jets = pd.concat([jets, cur_jet])

    return jets


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
                   columns=None):
    """Retrieve particle data stored in file, optionally based on where
    criteria.

    Keyword arguments:
    fname: (str) path to file storing the data (must be .h5)
    data_id: (str) id assigned to the data upon storage
    where: (list or None) Term (or convertible) objects, optional
    start: (int or None) event number to start selection
    stop: (int or None) event number to stop selection
    columns: (list or None) list of columns to return (None = all)
    """
    with pd.HDFStore(path=fname, mode='r') as store:
        return store.select(key, where, start, stop, columns, auto_close=True)


class HepData:
    def __init__(self, in_fname, interv):
        self.path = in_fname

        self.__data_obj = ReadHepmc.Hepmc(
                in_fname,
                start=interv[0],
                stop=interv[1])
        
        self.__evt_offset = interv[0]
        self.num_evts = self.__data_obj.n_events
        self.clustered = False


    def __is_event_set(self):
        return not (self.__data_obj.selected_index == None)

    def __get_event(self):
        return self.__data_obj.selected_index

    def __set_event(self, evt_num):
        self.__data_obj.selected_index = evt_num


    def event_isolate(func):
        """Returns a function passed, isolated from outer event setting.
        Note: final argument in function definition should be evt_num.
        """
        
        def isol_func(self, *args, **kwargs):
            if ('evt_num' in kwargs):
                evt_num = kwargs['evt_num']
            else:
                evt_num = args[-1]

            evt_context = self.__get_event()
            self.__set_event(evt_num)
            ret_val = func(self, *args, **kwargs)
            self.__set_event(evt_context)

            return ret_val
        return isol_func

            

    def _cluster(self, dR=0.8):
        self.dR = dR
        self.__jet_name = "MyJet"

        # calculate higher level (HL) data
        Components.add_all(self.__data_obj, inc_mass=True)

        # filter using the HL data
        filt_fin_pcl = FormJets.filter_ends # pcls at end of shower
        filt_observe = FormJets.filter_pt_eta # observable pt & eta range
        FormJets.create_jetInputs( # apply the filters to edit data inplace
                self.__data_obj,
                [filt_fin_pcl, filt_observe],
                batch_length=np.inf)

        # jet cluster events with anti-kt
        FormJets.cluster_multiapply(
                self.__data_obj,
                cluster_algorithm=FormJets.Traditional,
                dict_jet_params={'DeltaR': dR, 'ExpofPTMultiplier': -1},
                jet_name=self.__jet_name,
                batch_length=np.inf)

        self.clustered = True

    
    @event_isolate
    def _get_parent_idxs(self, tag_mcpid, evt_num):
        tag_idx = TrueTag.tag_particle_indices(
                self.__data_obj,
                tag_pids=tag_mcpid,
                include_antiparticles=False)
        return tag_idx


    @event_isolate
    def _get_lead_idx(self, pcl_idxs, evt_num):
        """Returns the particle with highest pT from a list of indices.
        """
        pt = self.__data_obj.PT[pcl_idxs]
        lead_idx = pcl_idxs[np.argmax(pt)]

        return lead_idx


    @event_isolate
    def _get_jet_idxs(self, tag_idxs, evt_num):
        """Get indices of jets associated with tag_idxs in evt_num.
        """
        jet_idx = TrueTag.allocate(self.__data_obj, self.__jet_name,
                                    tag_idxs, self.dR**2)

        return jet_idx


    @event_isolate
    def _get_jet_children(self, jet_idx, evt_num):
        # indices and numbers of pcls in this jet
        children = getattr( # jet indices of direct children
                self.__data_obj,
                self.__jet_name + "_Child1")[jet_idx]

        return children


    def _get_jet(self, tag_mcpid, jet_idx, parent_idx, evt_num):
        self.__data_obj.selected_index = evt_num

        # total num final state pcls in this event
        num_pcls_in_evt = len(self.__data_obj.JetInputs_Px)

        # indices and numbers of pcls in this jet
        children = self._get_jet_children(jet_idx, evt_num)
        constit_mask = children == -1 # mask for no children => final pcls
        
        num_pcls = np.sum(constit_mask) + 1

        # constructing a multi-index for output dataframe
        int_to_ext = { # defining the internal to external interface map
                'mcpid': 'MCPID',
                'energy': 'Energy',
                'px': 'Px',
                'py': 'Py',
                'pz': 'Pz'
                }

        # creating the empty dataframe structure
        parent_data = {}
        parent_data['parent'] = True

        for key, attr in int_to_ext.items():
            parent_data[key] = getattr(self.__data_obj, attr)[parent_idx]

        df_cols = list(parent_data.keys())

        df_idx = pd.Index( # indexes the event number for each particle
            np.full(num_pcls, evt_num + self.__evt_offset),
            name='event')
        df = pd.DataFrame([], index=df_idx, columns=df_cols)

        # print('num pcls are {}'.format(num_pcls))
        # print('parent data is

        # filling the dataframe with pcl data
        for key, attr in int_to_ext.items():
            if (key == 'mcpid'): # mcpids indexed per event, not per jet
                # get the indices of pcl and vertex constits within this jet
                pseudo_pcl_idxs = getattr(
                        self.__data_obj,
                        self.__jet_name + "_InputIdx")[jet_idx]

                # remove the pseudo from the list of indices
                # substitute for a mask where children == -1
                pcl_idxs = pseudo_pcl_idxs[pseudo_pcl_idxs < num_pcls_in_evt]

                # map indices from this jet to get idx wrt the whole event
                pcl_idxs = self.__data_obj.JetInputs_SourceIdx[pcl_idxs]

                # use as fancy index for event-wide MCPID listing
                mcpids = getattr(self.__data_obj, attr)
                data = np.array(mcpids[pcl_idxs], dtype=np.int32)

            else: # all other data can be accessed through same jet indexing
                attr_str = self.__jet_name + '_' + attr
                data = getattr(self.__data_obj,
                               attr_str)[jet_idx][constit_mask]
                data = np.array(data, dtype=np.float64)


            data = np.insert(data, 0, parent_data[key])
            df.loc[:, key] = data

        is_parent = np.full(num_pcls, False, dtype=np.bool_)
        is_parent[0] = True

        df.loc[:, 'parent'] = is_parent


        self.__data_obj.selected_index = None
        return df 
    

    def get_jet_constits(self, tag_mcpid):
        if (not self.clustered):
            self._cluster()

        # getting the jet from the first event
        evt = 0
        tag_idxs = self._get_parent_idxs(tag_mcpid=tag_mcpid, evt_num=evt)
        lead_idx = self._get_lead_idx(tag_idxs, evt)
        jet_idx = self._get_jet_idxs([lead_idx], evt)[0]
        lead_jets = self._get_jet(tag_mcpid, jet_idx, lead_idx, evt)

        # concatenating jets from subsequent events
        for evt in range(1, self.num_evts):
            tag_idxs = self._get_parent_idxs(tag_mcpid, evt)
            lead_idx = self._get_lead_idx(tag_idxs, evt)
            jet_idx = self._get_jet_idxs([lead_idx], evt)[0]

            lead_jets = pd.concat([
                lead_jets,
                self._get_jet(tag_mcpid, jet_idx, lead_idx, evt)
            ])

        return lead_jets
