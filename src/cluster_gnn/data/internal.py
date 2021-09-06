import h5py
from os.path import basename

class EventLoader:
    def __init__(self, path, key):
        self.path = path
        self.key = key
        self.__evt_iter = None
        self.__grp = None

    def __enter__(self):
        self.buffer = h5py.File(self.path, 'r')
        return self

    def __iter__(self):
        self.__evt_iter = iter(self.buffer[self.key])
        return self

    def __next__(self):
        grp_key = next(self.__evt_iter)
        self.__grp = self.buffer[self.key][grp_key]
        return self
    
    def __exit__(self, exc_type, exc_value, exc_traceback):
        self.buffer.close()

    def get_pmu(self, key='pmu'):
        return self.__grp[key][...]

    def get_signal(self, key='is_signal'):
        return self.__grp[key][...]

    def get_pdg(self):
        return self.__grp['pdg'][...]

    def get_custom(self, key):
        return self.__grp[key][...]

    def get_evt_name(self):
        return basename(self.__grp.name)

    def set_evt(self, evt_num):
        self.__evt_iter = None
        grp_key = f'event_{evt_num:06}'
        self.__grp = self.buffer[self.key][grp_key]
