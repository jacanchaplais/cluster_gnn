from functools import partial
from fractions import Fraction

import numpy as np
import pandas as pd

from cluster_gnn.data import PDG_DATA_PATH


def frac(num_str, obj_mode=False):
    """Converts string formatted fraction into number.

    Keyword arguments:
        num_str (str) -- string rep of rational number, eg. '1/2'
        obj_mod (bool) -- if True returns a fractions.Fraction object,
            if False returns a float

    example:
        In [1]: frac('1/2')
        Out [1]: 0.5

    Note on missing data:
        if passed empty string, will return 0,
        if passed '?', will return NaN
        other edge cases will raise a ValueError

    """
    if obj_mode == False:
        cast_frac = lambda inp: float(Fraction(inp))
    elif obj_mode == True:
        cast_frac = Fraction
    try:
        return cast_frac(num_str)
    except ValueError:
        if num_str == '':
            return cast_frac('0/1')
        elif num_str == '?':
            return np.nan

class LookupPDG:
    def __init__(self, frac_obj=False):
        frac_cols = ['I', 'G', 'P', 'C', 'Charge']
        cast_frac = partial(frac, obj_mode=frac_obj)
        converters = dict.fromkeys(frac_cols, cast_frac)
        lookup_table = pd.read_csv(
            PDG_DATA_PATH, sep=',', comment='#', converters=converters)
        self.__lookup = lookup_table.set_index('ID')

    def get_charge(self, pdgs):
        """Returns a Python list containing a mapping from each
        input pdg code to its associated charge.
        """
        return self.__lookup.loc[pdgs]['Charge'].to_list()
