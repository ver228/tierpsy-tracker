from functools import partial

from .processVideo import processVideo, isGoodVideo


def args_(fn, param):
    #step requirements
    requirements = [('can_read_video', partial(isGoodVideo, fn['original_video']))]
    if param.is_WT2:
        from ..compress_add_data import storeAdditionalDataSW, hasAdditionalFiles
        #if a shaffer single worm video does not have the additional files (info.xml log.csv) do not even execute the compression 
        requirements += [('has_additional_files', partial(hasAdditionalFiles, fn['original_video']))]

    #build input arguments for processVideo
    p = param.p_dict
    # getROIMask
    mask_param_f = ['mask_min_area', 'mask_max_area', 'thresh_block_size', 
        'thresh_C', 'dilation_size', 'keep_border_data', 'is_light_background']
    mask_param = {x.replace('mask_', ''):p[x] for x in mask_param_f}

    # bgnd subtraction
    bgnd_param_mask_f = ['mask_bgnd_buff_size', 'mask_bgnd_frame_gap', 'is_light_background']
    bgnd_param_mask = {x.replace('mask_bgnd_', ''):p[x] for x in bgnd_param_mask_f}
    
    if bgnd_param_mask['buff_size']<=0 or bgnd_param_mask['frame_gap']<=0:
        bgnd_param_mask = {}
    
    # FOV splitting
    fovsplitter_param_f = ['MWP_total_n_wells', 'MWP_whichsideup', 'MWP_well_shape']
    if not all(k in p for k in fovsplitter_param_f):
        fovsplitter_param = {}
    else:
        fovsplitter_param = {x.replace('MWP_',''):p[x] for x in fovsplitter_param_f}   
    
    if fovsplitter_param['total_n_wells']<0:
        fovsplitter_param = {}
    
    compress_vid_param = {
            'buffer_size': p['compression_buff'],
            'save_full_interval': p['save_full_interval'],
            'mask_param': mask_param,
            'bgnd_param': bgnd_param_mask,
            'expected_fps': p['expected_fps'],
            'microns_per_pixel' : p['microns_per_pixel'],
            'is_extract_timestamp': p['is_extract_timestamp'],
            'fovsplitter_param': fovsplitter_param,
        }

    argkws_d = {
            'video_file': fn['original_video'], 
            'masked_image_file' : fn['masked_image'],
            'compress_vid_param' : compress_vid_param
        }

    #arguments used by AnalysisPoints.py
    args = {
        'func':processVideo,
        'argkws' : argkws_d,
        'input_files' : [fn['original_video']],
        'output_files': [fn['masked_image']],
        'requirements' : requirements,
        }

    return args