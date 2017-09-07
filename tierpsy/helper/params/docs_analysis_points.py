'''
Dictionary of the analysis points that will be executed for a given analysis_point.
If the point is not in the dictionary the points used will be the ones in DEFAULT.
'''
valid_analysis_points = ['WORM', 'WORM_RIG', 'WT2', 'PHARYNX', 'ZEBRAFISH', 'MANUAL', 'TEST']

dflt_analysis_points = {
    'DEFAULT':
    ['COMPRESS',
    'VID_SUBSAMPLE',
    'TRAJ_CREATE',
    'TRAJ_JOIN',
    'SKE_INIT',
    'BLOB_FEATS',
    'SKE_CREATE',
    'SKE_FILT',
    'SKE_ORIENT',
    'INT_PROFILE',
    'INT_SKE_ORIENT',
    'FEAT_CREATE'
    ],

    'WORM_RIG':
    ['COMPRESS',
    'VID_SUBSAMPLE',
    'TRAJ_CREATE',
    'TRAJ_JOIN',
    'SKE_INIT',
    'BLOB_FEATS',
    'SKE_CREATE',
    'SKE_FILT',
    'SKE_ORIENT',
    'INT_PROFILE',
    'INT_SKE_ORIENT',
    'FEAT_CREATE',
    'FEAT_FOOD'
    ],


    'WT2' : 
    ['COMPRESS',
    'COMPRESS_ADD_DATA',
    'VID_SUBSAMPLE',
    'TRAJ_CREATE',
    'TRAJ_JOIN',
    'SKE_INIT',
    'BLOB_FEATS',
    'SKE_CREATE',
    'SKE_FILT',
    'SKE_ORIENT',
    'STAGE_ALIGMENT',
    'INT_PROFILE',
    'INT_SKE_ORIENT',
    'FEAT_CREATE',
    'WCON_EXPORT'
    ],

    'MANUAL':
    ['FEAT_MANUAL_CREATE'],

    'TEST':
    ['COMPRESS',
    'VID_SUBSAMPLE',
    'TRAJ_CREATE',
    'TRAJ_JOIN',
    'SKE_INIT',
    'BLOB_FEATS',
    'SKE_CREATE',
    'SKE_FILT',
    'SKE_ORIENT',
    'INT_PROFILE',
    'INT_SKE_ORIENT',
    'FEAT_TIERPSY'
    ],
}
