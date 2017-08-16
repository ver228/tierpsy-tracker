import os
import tables
from .misc import TABLE_FILTERS

RESERVED_EXT = ['_skeletons.hdf5', 
                '_trajectories.hdf5', 
                '_features.hdf5', 
                '_intensities.hdf5', 
                '_feat_manual.hdf5',
                '_subsample.avi',
                '.wcon.zip',
                '_featuresN.hdf5'
                ]

def remove_ext(fname):
    for rext in RESERVED_EXT:
        if fname.endswith(rext):
            return fname.replace(rext, '')
    return os.path.splitext(fname)[0]

def get_base_name(fname):
    return os.path.basename(remove_ext(fname))

def replace_subdir(original_dir, original_subdir, new_subdir):
    # construct the results dir on base of the mask_dir_root
    original_dir = os.path.normpath(original_dir)
    subdir_list = original_dir.split(os.sep)
    for ii in range(len(subdir_list))[::-1]:
        if subdir_list[ii] == original_subdir:
            subdir_list[ii] = new_subdir
            break
    # the counter arrived to zero, add new_subdir at the end of the directory
    if ii == 0:
        if subdir_list[-1] == '':
            del subdir_list[-1]
        subdir_list.append(new_subdir)

    return (os.sep).join(subdir_list)

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


