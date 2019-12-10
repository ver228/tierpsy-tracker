'''
Some functions share between different files that make 
more sense to group together than in other files.
'''
import textwrap
from collections import OrderedDict

def get_prefix_params(input_params, prefix):
    return {x.replace(prefix, ''):input_params[x] for x in input_params if x.startswith(prefix)}

def repack_dflt_list(dflt_list, valid_options={}):
    '''
        Reformat the variable text_info to make it more readable.
        dflt_list - is a list of tuples in the format
                    [(name, default_value, text_info), ...]
        
        valid_options - is a dictionary that contains as dictionary keys the features that
                        are allowed to have specific values, and as dictionary values a list
                        with the allowed values.
    '''
    def _format_var_info(input_tuple):
        
        name, dftl_val, info_txt = input_tuple


        if name in valid_options:
            info_txt += ' Valid_options ({})'.format(','.join([str(vo) for vo in valid_options[name]]))
        info_txt = textwrap.dedent(info_txt)
        info_txt = textwrap.fill(info_txt)
        return name, dftl_val, info_txt

    dflt_list = list(map(_format_var_info, dflt_list))

    #separate parameters default data into dictionaries for values and help
    values_dict = OrderedDict()
    info_dict = OrderedDict()
    for name, dflt_value, info in dflt_list:
        values_dict[name] = dflt_value
        info_dict[name] = info

    return values_dict, info_dict