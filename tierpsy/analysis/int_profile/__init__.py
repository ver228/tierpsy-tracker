from .getIntensityProfile import getIntensityProfile

def args_(fn, param):
    
    p = param.p_dict
    argkws_d = {
        'masked_image_file': fn['masked_image'], 
        'skeletons_file': fn['skeletons'],
        'intensities_file': fn['intensities'],
        'width_resampling': p['int_width_resampling'],
        'length_resampling': p['int_length_resampling'],
        'min_num_skel': -1,
        'smooth_win': 11,
        'pol_degree': 3,
        'width_percentage': p['int_avg_width_frac'],
        'save_maps': p['int_save_maps']
        }

    requirements = ['SKE_CREATE']
    main_func = getIntensityProfile
    
    if param.is_WT2:
        from functools import partial
        from ..contour_orient import ventral_orient_wrapper
        main_func = partial(ventral_orient_wrapper, main_func, fn['skeletons'], param.p_dict['ventral_side'])
        main_func.__name__ = getIntensityProfile.__name__  # I use the name for provenance tracking

    #arguments used by AnalysisPoints.py
    return {
        'func': main_func,
        'argkws': argkws_d,
        'input_files' : [fn['skeletons'],fn['masked_image']],
        'output_files': [fn['intensities'], fn['skeletons']],
        'requirements' : ['SKE_CREATE'],
    }