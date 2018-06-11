
deprecated_analysis_alias = {'WORM':'OPENWORM', 'WORM_SINGLE':'OPENWORM', 'WT2':'OPENWORM_WT2', 'MANUAL':'OPENWORM_MANUAL', 'TEST':'TIERPSY_AEX'}
dlft_analysis_type = 'TIERPSY'

'''
Dictionary of the analysis points that will be executed for a given analysis_point.
If the point is not in the dictionary the points used will be the ones in DEFAULT.
'''

dflt_analysis_points = {
    'BASE' : 
        ['COMPRESS',
        'TRAJ_CREATE',
        'TRAJ_JOIN',
        'SKE_INIT',
        'BLOB_FEATS',
        'SKE_CREATE',
        'SKE_FILT',
        'SKE_ORIENT',
        'INT_PROFILE',
        'INT_SKE_ORIENT'
        ],
    'BASE_WT2' : 
        ['COMPRESS',
        'COMPRESS_ADD_DATA',
        'TRAJ_CREATE',
        'TRAJ_JOIN',
        'SKE_INIT',
        'BLOB_FEATS',
        'SKE_CREATE',
        'SKE_FILT',
        'SKE_ORIENT',
        'STAGE_ALIGMENT',
        'INT_PROFILE',
        'INT_SKE_ORIENT'
        ],
    
    'TIERPSY_FEATURES' : 
        ['FEAT_INIT',
        'FEAT_TIERPSY'
        ],
    
    'OPENWORM_FEATURES' : ['FEAT_CREATE'],
    'OPENWORM_MANUAL' : ['FEAT_MANUAL_CREATE'],

    'PHARYNX' : 
        ['COMPRESS',
        'TRAJ_CREATE',
        'TRAJ_JOIN',
        'SKE_INIT',
        'BLOB_FEATS',
        'SKE_CREATE',
        'SKE_PHARYNX'
        ]
    }

#get full sequence for openwom features
dflt_analysis_points['OPENWORM'] = dflt_analysis_points['BASE'] + dflt_analysis_points['OPENWORM_FEATURES']
dflt_analysis_points['OPENWORM_WT2'] = dflt_analysis_points['BASE_WT2'] + dflt_analysis_points['OPENWORM_FEATURES'] + ['WCON_EXPORT']

#get full sequence for tierpsy features
dflt_analysis_points['TIERPSY'] = dflt_analysis_points['BASE'] + dflt_analysis_points['TIERPSY_FEATURES']
dflt_analysis_points['TIERPSY_WT2'] = dflt_analysis_points['BASE_WT2'] + dflt_analysis_points['TIERPSY_FEATURES']

#the contour calculation is only valid for andrew's lab
dflt_analysis_points['TIERPSY_AEX'] = dflt_analysis_points['BASE'] + ['FOOD_CNT'] + dflt_analysis_points['TIERPSY_FEATURES']

#adding _SINGLE to analysis_type will force the data to be treated as a single worm
dflt_analysis_points['OPENWORM_SINGLE'] = dflt_analysis_points['OPENWORM']
dflt_analysis_points['TIERPSY_SINGLE'] = dflt_analysis_points['TIERPSY']

#only valid analysis points
valid_analysis_types = list(sorted(dflt_analysis_points.keys()))



