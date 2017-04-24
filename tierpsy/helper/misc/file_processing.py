import os
import tables
from .misc import TABLE_FILTERS

RESERVED_EXT = ['_skeletons.hdf5', 
                '_trajectories.hdf5', 
                '_features.hdf5', 
                '_intensities.hdf5', 
                '_feat_manual.hdf5',
                '_subsampled.avi',
                '.wcon.zip']

def get_base_name(fname):
    bn = os.path.basename(fname)
    for rext in RESERVED_EXT:
        if bn.endswith(rext):
            return bn.replace(rext, '')

    return os.path.splitext(bn)[0]


def save_modified_table(file_name, modified_table, table_name):
    tab_recarray = modified_table.to_records(index=False)
    with tables.File(file_name, "r+") as fid:
        dum_name = table_name + '_d'
        if '/' + dum_name in fid:
            fid.remove_node('/', dum_name)

        newT = fid.create_table(
            '/',
            dum_name,
            obj=tab_recarray,
            filters=TABLE_FILTERS)

        oldT = fid.get_node('/' + table_name)
        old_args = [x for x in dir(oldT._v_attrs) if not x.startswith('_')]
        for key in old_args:
            
            if not key in newT._v_attrs and not key.startswith('FIELD'):
                newT.attrs[key] = oldT.attrs[key]
                
        fid.remove_node('/', table_name)
        newT.rename(table_name)


